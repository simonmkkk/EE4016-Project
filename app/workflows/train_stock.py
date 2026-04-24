from __future__ import annotations

import argparse
import glob
import hashlib
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from torch import nn

from ..constants import INTERVAL_ID_ORDER, INTERVAL_ID_UNKNOWN
from ..domain.datasets import infer_symbol_from_path, read_training_frame
from ..infra.files import dump_json, dump_pickle, load_json
from ..model import LSTMDir, lstm_feature_columns
from ..paths import MODEL_DIR
from ..sequences import build_seq_multi, evaluate_split, val_probs_for_threshold_search


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LINE_WIDTH = 70
INTERVAL_EMBED_DIM = 8
NUM_INTERVAL_EMBEDDINGS = INTERVAL_ID_UNKNOWN + 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvs", nargs="*", help="paths to one or more CSV files")
    parser.add_argument("--csv_dir", help="directory containing CSV files to load")
    parser.add_argument(
        "--ticker",
        help="filter CSVs by ticker when using --csv_dir (e.g. AAPL); also sets model/{ticker}/ output folder",
    )
    parser.add_argument("--save_model", required=True)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--use_attn", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument(
        "--eval_threshold",
        type=float,
        default=0.5,
        help="Base decision threshold; also grid fallback when val search is disabled",
    )
    parser.add_argument(
        "--threshold_objective",
        type=str,
        default="f1_macro",
        choices=["f1_pos", "f1_macro", "balanced_accuracy", "mcc"],
        help="Objective used by val threshold search.",
    )
    drift_group = parser.add_mutually_exclusive_group()
    drift_group.add_argument(
        "--threshold-drift-adjust",
        dest="threshold_drift_adjust",
        action="store_true",
        help="Adjust selected val threshold by train/val probability drift.",
    )
    drift_group.add_argument(
        "--no-threshold-drift-adjust",
        dest="threshold_drift_adjust",
        action="store_false",
        help="Disable probability drift adjustment for threshold.",
    )
    parser.set_defaults(threshold_drift_adjust=True)
    threshold_group = parser.add_mutually_exclusive_group()
    threshold_group.add_argument(
        "--val-threshold-search",
        dest="val_threshold_search",
        action="store_true",
        help="Pick threshold on val to maximize F1 (default)",
    )
    threshold_group.add_argument(
        "--no-val-threshold-search",
        dest="val_threshold_search",
        action="store_false",
        help="Use fixed --eval_threshold for val/test metrics",
    )
    parser.set_defaults(val_threshold_search=True)
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of LSTM layers (residual between layer 2..N)",
    )
    parser.add_argument(
        "--lr_schedule_patience",
        type=int,
        default=5,
        help="ReduceLROnPlateau patience (epochs without val loss improvement)",
    )
    parser.add_argument(
        "--walk_forward_folds",
        type=int,
        default=0,
        help="Optional rolling evaluation folds on test split",
    )
    parser.add_argument(
        "--early_stop_metric",
        type=str,
        default="val_loss",
        choices=["val_loss", "val_f1"],
        help="Early stopping monitor.",
    )
    parser.add_argument("--protocol", type=str, help="Path to experiment protocol json")
    parser.add_argument(
        "--no-pos-weight",
        "--no_pos_weight",
        action="store_true",
        help="Unweighted BCEWithLogitsLoss (diagnostic).",
    )
    parser.add_argument(
        "--pos-weight-min",
        type=float,
        default=None,
        metavar="W",
        help="Clamp balanced pos_weight=max(neg/pos, W).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="Dropout probability applied before the FC layer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="L2 weight decay for Adam optimiser.",
    )
    parser.add_argument(
        "--label_threshold",
        type=float,
        default=0.0,
        metavar="T",
        help="Minimum log-return to label a bar as UP.",
    )
    parser.add_argument(
        "--label_threshold_quantile",
        type=float,
        default=None,
        metavar="Q",
        help="Per-series adaptive label threshold.",
    )
    return parser


def _interactive_args() -> list[str]:
    print("\n=== Train - Interactive mode ===")
    csv_mode = input("Load from directory or list files? (d=directory / f=files) [d] ").strip().lower()
    args: list[str] = []
    if csv_mode.startswith("f"):
        csvs_raw = input("Enter CSV paths (space-separated): ").strip()
        csvs = [p.strip().strip('"').strip("'") for p in csvs_raw.split()]
        args += ["--csvs", *csvs]
    else:
        csv_dir = input("Directory path (contains *.csv): ").strip().strip('"').strip("'") or "."
        args += ["--csv_dir", csv_dir]

    model_name = input("Output model filename (e.g. model.pt) [model.pt] ").strip() or "model.pt"
    if not model_name.endswith(".pt"):
        model_name += ".pt"
    args += ["--save_model", model_name]

    window = input("window length [100] ").strip() or "100"
    epochs = input("epochs [40] ").strip() or "40"
    args += ["--window", window, "--epochs", epochs]
    if input("Use attention? (y/n) [n] ").strip().lower().startswith("y"):
        args.append("--use_attn")
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    cli_args = list(argv) if argv is not None else None
    if cli_args is None:
        cli_args = sys.argv[1:]
    if not cli_args:
        cli_args = _interactive_args()
    args = build_parser().parse_args(cli_args)
    args.command_argv = cli_args
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_frame_by_ratio(
    frame: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_rows = len(frame)
    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))
    return (
        frame.iloc[:train_end].copy(),
        frame.iloc[train_end:val_end].copy(),
        frame.iloc[val_end:].copy(),
    )


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_commit_hash() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def walk_forward_metrics(
    model: nn.Module,
    test_frames: list[pd.DataFrame],
    feats_lstm: list[str],
    window: int,
    device: torch.device,
    threshold: float,
    folds: int,
    *,
    use_interval_embedding: bool,
    inference_batch_size: int = 256,
) -> list[dict]:
    if folds <= 1:
        return []
    per_fold: list[dict] = []
    for series_index, frame in enumerate(test_frames):
        n_rows = len(frame)
        if n_rows <= window + folds:
            continue
        step = n_rows // folds
        for fold_index in range(folds):
            start = fold_index * step
            end = n_rows if fold_index == folds - 1 else (fold_index + 1) * step
            chunk = frame.iloc[start:end].copy()
            metrics = evaluate_split(
                model,
                [chunk],
                feats_lstm,
                window,
                device,
                threshold,
                use_interval_embedding=use_interval_embedding,
                inference_batch_size=inference_batch_size,
            )
            if metrics is not None:
                metrics["series_index"] = series_index
                metrics["fold_index"] = fold_index
                per_fold.append(metrics)
    return per_fold


def print_kv_section(title: str, rows: list[tuple[str, str]], border: str = "=") -> None:
    print("\n" + border * LINE_WIDTH)
    print(title)
    print(border * LINE_WIDTH)
    for key, value in rows:
        print(f"  {key:<14}: {value}")
    print(border * LINE_WIDTH)


def print_eval_section(
    val_metrics: dict | None,
    test_metrics: dict | None,
    window: int,
    val_thr_info: str | None = None,
) -> None:
    print("\n" + "=" * LINE_WIDTH)
    print("FINAL EVALUATION")
    print("=" * LINE_WIDTH)
    if val_thr_info:
        print(f"  {val_thr_info}")
        print("-" * LINE_WIDTH)
    if val_metrics is None and test_metrics is None:
        print(f"  [WARN] val/test rows are insufficient for window={window}")
        print("=" * LINE_WIDTH)
        return

    horiz = "─"
    vert = "│"
    widths = [6, 7, 10, 10, 10, 10, 6]

    def top(cols: list[int]) -> str:
        return "  ┌" + "┬".join(horiz * (w + 2) for w in cols) + "┐"

    def mid(cols: list[int]) -> str:
        return "  ├" + "┼".join(horiz * (w + 2) for w in cols) + "┤"

    def bot(cols: list[int]) -> str:
        return "  └" + "┴".join(horiz * (w + 2) for w in cols) + "┘"

    def row(cells: list[object], cols: list[int]) -> str:
        parts = [f" {str(v).center(w)} " for v, w in zip(cells, cols)]
        return "  " + vert + vert.join(parts) + vert

    print(top(widths))
    print(row(["Split", "N", "Accuracy", "Precision", "Recall", "F1", "Thr"], widths))
    print(mid(widths))

    def metrics_row(name: str, metrics: dict | None) -> str:
        if metrics is None:
            return row([name, "N/A", "-", "-", "-", "-", "-"], widths)
        return row(
            [
                name,
                f"{metrics['n_samples']:,}",
                f"{metrics['accuracy']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1']:.4f}",
                f"{metrics['threshold']:.2f}",
            ],
            widths,
        )

    print(metrics_row("VAL", val_metrics))
    print(metrics_row("TEST", test_metrics))
    print(bot(widths))

    if val_metrics is None:
        print(f"  [WARN] val rows are insufficient for window={window}")
    if test_metrics is None:
        print(f"  [WARN] test rows are insufficient for window={window}")

    conf_widths = [6, 6, 6, 6, 6, 12]
    print()
    print(top(conf_widths))
    print(row(["Split", "TN", "FP", "FN", "TP", "P(pred=UP)"], conf_widths))
    print(mid(conf_widths))

    def conf_row(name: str, metrics: dict | None) -> str:
        if metrics is None or "tn" not in metrics:
            return row([name, "-", "-", "-", "-", "-"], conf_widths)
        return row(
            [
                name,
                metrics["tn"],
                metrics["fp"],
                metrics["fn"],
                metrics["tp"],
                f"{metrics['pred_positive_rate']:.1%}",
            ],
            conf_widths,
        )

    print(conf_row("VAL", val_metrics))
    print(conf_row("TEST", test_metrics))
    print(bot(conf_widths))
    print("=" * LINE_WIDTH)


def print_training_header(
    symbol: str,
    csv_list: list[str],
    save_path: Path,
    args: argparse.Namespace,
) -> None:
    test_ratio = 1 - args.train_ratio - args.val_ratio
    print_kv_section(
        "TRAINING CONFIGURATION",
        [
            ("ticker", symbol),
            ("csv count", str(len(csv_list))),
            ("save path", str(save_path)),
            ("window", str(args.window)),
            ("epochs", str(args.epochs)),
            ("batch", str(args.batch)),
            ("lr", str(args.lr)),
            ("patience", str(args.patience)),
            ("use_attn", str(args.use_attn)),
            ("num_layers", str(args.num_layers)),
            ("eval_threshold (base)", str(args.eval_threshold)),
            ("val_threshold_search", str(args.val_threshold_search)),
            ("thr_objective", str(args.threshold_objective)),
            ("thr_drift_adjust", str(args.threshold_drift_adjust)),
            ("lr_sched_patience", str(args.lr_schedule_patience)),
            ("early_stop_metric", str(args.early_stop_metric)),
            ("no_pos_weight", str(args.no_pos_weight)),
            (
                "pos_weight_min",
                str(args.pos_weight_min) if args.pos_weight_min is not None else "-",
            ),
            ("dropout", f"{args.dropout:.2f}"),
            ("weight_decay", f"{args.weight_decay:.2e}"),
            (
                "label_threshold",
                (
                    f"quantile={args.label_threshold_quantile:.2f} (per-series adaptive)"
                    if args.label_threshold_quantile is not None
                    else (
                        f"{args.label_threshold:.4f} (any positive)"
                        if args.label_threshold == 0.0
                        else f"{args.label_threshold:.4f} (log_ret > {args.label_threshold:.4f})"
                    )
                ),
            ),
            (
                "split ratios",
                f"train={args.train_ratio:.2f} val={args.val_ratio:.2f} test={test_ratio:.2f}",
            ),
        ],
        border="=",
    )
    print("  input csvs")
    for path in csv_list:
        print(f"    - {path}")
    print("-" * LINE_WIDTH)


def _resolve_csv_list(args: argparse.Namespace) -> list[str]:
    csv_list = list(args.csvs or [])
    if args.csv_dir:
        all_csv = glob.glob(os.path.join(args.csv_dir, "*.csv"))
        if args.ticker:
            ticker_upper = args.ticker.strip().upper()
            csv_list += [
                path
                for path in all_csv
                if Path(path).stem.upper().startswith(ticker_upper + "_")
            ]
            if not csv_list:
                raise SystemExit(
                    f"[ERROR] No CSV in {args.csv_dir!r} for ticker {ticker_upper} "
                    f"(e.g. {ticker_upper}_1d_10y.csv)"
                )
        else:
            csv_list += all_csv
    if not csv_list:
        raise SystemExit("[ERROR] Must specify --csvs or --csv_dir")
    return csv_list


def run_training(args: argparse.Namespace) -> int:
    protocol = load_json(args.protocol) if args.protocol else {}
    if protocol:
        split_cfg = protocol.get("split", {})
        if split_cfg.get("method") == "ratio":
            args.train_ratio = float(split_cfg.get("train_ratio", args.train_ratio))
            args.val_ratio = float(split_cfg.get("val_ratio", args.val_ratio))

    if not (
        0 < args.train_ratio < 1
        and 0 < args.val_ratio < 1
        and args.train_ratio + args.val_ratio < 1
    ):
        raise SystemExit("[ERROR] train_ratio and val_ratio must be in (0,1), and sum < 1")
    if not (0 < args.eval_threshold < 1):
        raise SystemExit("[ERROR] eval_threshold must be in (0,1)")
    if args.num_layers < 1:
        raise SystemExit("[ERROR] num_layers must be >= 1")
    if args.walk_forward_folds < 0:
        raise SystemExit("[ERROR] walk_forward_folds must be >= 0")
    if args.pos_weight_min is not None and args.pos_weight_min <= 0:
        raise SystemExit("[ERROR] --pos-weight-min must be > 0")
    if args.no_pos_weight and args.pos_weight_min is not None:
        print("[WARN] --no-pos-weight set; --pos-weight-min ignored")

    set_seed(args.seed)
    csv_list = _resolve_csv_list(args)
    symbol = args.ticker.strip().upper() if args.ticker else infer_symbol_from_path(csv_list[0])
    model_dir = MODEL_DIR / symbol
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / (Path(args.save_model).name or "model.pt")
    print_training_header(symbol, csv_list, save_path, args)

    empty_paths: list[str] = []
    frames: list[pd.DataFrame] = []
    for path in csv_list:
        frame = read_training_frame(
            path,
            label_threshold=args.label_threshold,
            label_threshold_quantile=args.label_threshold_quantile,
        )
        if len(frame) == 0:
            empty_paths.append(path)
        else:
            frames.append(frame)

    if empty_paths:
        print(
            f"[WARN] Skipped {len(empty_paths)} CSV(s) with no rows after feature engineering "
            "(series too short for indicators + next-bar target):"
        )
        for path in empty_paths:
            print(f"    - {path}")
    if not frames:
        raise SystemExit(
            "[ERROR] No usable rows after feature engineering. "
            "Use longer histories or fewer intraday intervals."
        )

    feats = [
        col
        for col in frames[0].columns
        if col not in ["date", "log_ret", "direction", "series_id", "_label_thr_used"]
    ]
    if "granularity" not in feats:
        feats.append("granularity")
    if "interval_id" not in feats:
        feats.append("interval_id")

    feats_scaled = [col for col in feats if col != "interval_id"]
    use_interval_embedding = "interval_id" in feats
    feats_lstm, _ = lstm_feature_columns(feats, use_interval_embedding)

    split_triplets = [
        split_frame_by_ratio(frame, args.train_ratio, args.val_ratio)
        for frame in frames
    ]
    train_frames = [triplet[0] for triplet in split_triplets if len(triplet[0]) > 0]
    val_frames = [triplet[1] for triplet in split_triplets if len(triplet[1]) > 0]
    test_frames = [triplet[2] for triplet in split_triplets if len(triplet[2]) > 0]
    if not train_frames:
        raise SystemExit("[ERROR] Train split is empty for all series")

    scaler = StandardScaler()
    train_stack = pd.concat(train_frames, ignore_index=True)
    scaler.fit(train_stack[feats_scaled])
    for frame in train_frames:
        frame[feats_scaled] = scaler.transform(frame[feats_scaled]).astype(np.float32)
    for frame in val_frames:
        frame[feats_scaled] = scaler.transform(frame[feats_scaled]).astype(np.float32)
    for frame in test_frames:
        frame[feats_scaled] = scaler.transform(frame[feats_scaled]).astype(np.float32)

    X, y, iv_train = build_seq_multi(
        train_frames,
        feats_lstm,
        args.window,
        use_interval_embedding=use_interval_embedding,
    )
    if X.shape[0] == 0:
        raise SystemExit(f"[ERROR] Train sequences are insufficient for window={args.window}")
    X_val, y_val, iv_val = build_seq_multi(
        val_frames,
        feats_lstm,
        args.window,
        use_interval_embedding=use_interval_embedding,
    )
    if X_val.shape[0] == 0:
        raise SystemExit(f"[ERROR] Validation sequences are insufficient for window={args.window}")

    test_seq_count = build_seq_multi(
        test_frames,
        feats_lstm,
        args.window,
        use_interval_embedding=use_interval_embedding,
    )[0].shape[0]
    train_all_labels = np.concatenate([frame["direction"].values for frame in train_frames])
    train_pos_ratio = float(np.mean(train_all_labels))
    if args.label_threshold_quantile is not None:
        label_thr_note = (
            f"per-series quantile {args.label_threshold_quantile:.2f} "
            f"(top {1 - args.label_threshold_quantile:.0%} UP per series; "
            f"overall {train_pos_ratio:.1%} UP in train)"
        )
    elif args.label_threshold > 0.0:
        label_thr_note = (
            f"log_ret > {args.label_threshold:.4f} ({train_pos_ratio:.1%} UP in train)"
        )
    else:
        label_thr_note = f"log_ret > 0 ({train_pos_ratio:.1%} UP in train)"

    print_kv_section(
        "TRAINING DATA SUMMARY",
        [
            ("features", str(len(feats))),
            ("lstm_inputs", str(len(feats_lstm))),
            ("interval_embedding", str(use_interval_embedding)),
            ("label rule", label_thr_note),
            ("train rows", str(sum(len(frame) for frame in train_frames))),
            ("val rows", str(sum(len(frame) for frame in val_frames))),
            ("test rows", str(sum(len(frame) for frame in test_frames))),
            ("train seq", str(X.shape[0])),
            ("val seq", str(X_val.shape[0])),
            ("test seq", str(test_seq_count)),
        ],
        border="-",
    )
    if args.label_threshold_quantile is not None:
        print("  per-series label thresholds (quantile mode):")
        for frame in frames:
            series_id = frame["series_id"].iloc[0] if "series_id" in frame.columns else "?"
            threshold_used = (
                float(frame["_label_thr_used"].iloc[0])
                if "_label_thr_used" in frame.columns
                else float("nan")
            )
            positive_rate = float(frame["direction"].mean())
            print(f"    {series_id:<28}  thr={threshold_used:+.6f}  UP={positive_rate:.1%}")

    if iv_train is not None:
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X),
            torch.tensor(iv_train),
            torch.tensor(y),
        )
    else:
        dataset = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        args.batch,
        shuffle=True,
        generator=loader_gen,
    )

    model = LSTMDir(
        len(feats_lstm),
        att=args.use_attn,
        num_layers=args.num_layers,
        num_intervals=(NUM_INTERVAL_EMBEDDINGS if use_interval_embedding else None),
        embed_dim=INTERVAL_EMBED_DIM,
        dropout=args.dropout,
    ).to(DEVICE)

    pos_ratio = float(np.mean(y))
    neg_ratio = float(1.0 - pos_ratio)
    safe_pos = max(pos_ratio, 1e-6)
    balanced_pw = neg_ratio / safe_pos
    if args.no_pos_weight:
        criterion = nn.BCEWithLogitsLoss()
        print(
            f"[INFO] pos_ratio={pos_ratio:.4f} neg_ratio={neg_ratio:.4f} "
            f"balanced_pos_weight={balanced_pw:.4f} effective=unweighted (--no-pos-weight)"
        )
    else:
        pos_weight_val = balanced_pw
        if args.pos_weight_min is not None:
            pos_weight_val = max(pos_weight_val, float(args.pos_weight_min))
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32, device=DEVICE)
        )
        clamp_note = ""
        if args.pos_weight_min is not None and pos_weight_val > balanced_pw + 1e-12:
            clamp_note = f" (clamped from {balanced_pw:.4f} by --pos-weight-min)"
        print(
            f"[INFO] pos_ratio={pos_ratio:.4f} neg_ratio={neg_ratio:.4f} "
            f"balanced_pos_weight={balanced_pw:.4f} pos_weight={pos_weight_val:.4f}{clamp_note}"
        )

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.lr_schedule_patience,
    )
    val_xb = torch.tensor(X_val).to(DEVICE)
    val_yb = torch.tensor(y_val).to(DEVICE)
    val_ivb = torch.tensor(iv_val).to(DEVICE) if iv_val is not None else None

    best = 0.0 if args.early_stop_metric == "val_f1" else math.inf
    wait = 0
    epoch_history: list[dict] = []
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        if iv_train is not None:
            for xb, ivb, yb in data_loader:
                xb, ivb, yb = xb.to(DEVICE), ivb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(xb, ivb), yb)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(xb)
        else:
            for xb, yb in data_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(xb)
        avg_train = loss_sum / len(dataset)

        model.eval()
        with torch.no_grad():
            val_logits = model(val_xb, val_ivb) if val_ivb is not None else model(val_xb)
            avg_val = criterion(val_logits, val_yb).item()
            if args.early_stop_metric == "val_f1":
                val_probs = torch.sigmoid(val_logits).cpu().numpy().ravel()
                early_stop_val = float(
                    f1_score(
                        np.asarray(y_val).ravel(),
                        (val_probs > 0.5).astype(int),
                        zero_division=0,
                    )
                )
                improved = early_stop_val > best
            else:
                early_stop_val = avg_val
                improved = early_stop_val < best
        if improved:
            best, wait = early_stop_val, 0
            torch.save(model.state_dict(), save_path)
        else:
            wait += 1

        scheduler.step(avg_val)
        epoch_entry: dict = {
            "epoch": epoch + 1,
            "epochs": args.epochs,
            "train_loss": float(avg_train),
            "val_loss": float(avg_val),
            "early_stop_metric": args.early_stop_metric,
            "wait": int(wait),
            "patience": int(args.patience),
            "improved": bool(improved),
        }
        if args.early_stop_metric == "val_f1":
            epoch_entry["val_f1"] = float(early_stop_val)
            epoch_entry["best_val_f1"] = float(best)
            print(
                f"[E{epoch+1:03d}/{args.epochs:03d}] "
                f"train={avg_train:.4f} val_loss={avg_val:.4f} "
                f"val_f1={early_stop_val:.4f} best_f1={best:.4f} "
                f"wait={wait}/{args.patience}"
            )
        else:
            epoch_entry["best_val_loss"] = float(best)
            print(
                f"[E{epoch+1:03d}/{args.epochs:03d}] "
                f"train={avg_train:.4f} val={avg_val:.4f} "
                f"best={best:.4f} wait={wait}/{args.patience}"
            )
        epoch_history.append(epoch_entry)

        if not improved and wait >= args.patience:
            print(f"[STOP] Early stopping at epoch {epoch + 1} (patience={args.patience})")
            break
    print(f"[OK] model saved: {save_path}")

    scaler_path = save_path.with_suffix(".scaler.pkl")
    meta_path = save_path.with_suffix(".meta.json")
    dump_pickle(scaler_path, scaler)

    best_model = LSTMDir(
        len(feats_lstm),
        att=args.use_attn,
        num_layers=args.num_layers,
        num_intervals=(NUM_INTERVAL_EMBEDDINGS if use_interval_embedding else None),
        embed_dim=INTERVAL_EMBED_DIM,
        dropout=args.dropout,
    ).to(DEVICE)
    best_model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))

    effective_thr = float(args.eval_threshold)
    threshold_search_info: dict | None = None
    if args.val_threshold_search:
        cand = val_probs_for_threshold_search(
            best_model,
            val_frames,
            feats_lstm,
            args.window,
            DEVICE,
            use_interval_embedding=use_interval_embedding,
            inference_batch_size=args.batch,
        )
        if cand is not None:
            probs_v, y_v = cand
            y_rate = float(np.mean(y_v))
            lo, hi = float(np.min(probs_v)), float(np.max(probs_v))
            thr_uniform = np.arange(0.02, 0.991, 0.02)
            thr_pct = np.percentile(probs_v, np.arange(3, 100, 2))
            parts: list[np.ndarray] = [thr_uniform, thr_pct]
            if hi > lo + 1e-12:
                parts.append(np.linspace(lo, hi, num=min(64, max(8, int(len(probs_v) * 2)))))
            cand_ts = np.unique(np.clip(np.concatenate(parts), 1e-9, 1.0 - 1e-9))

            def prevalence_threshold(rate: float) -> float:
                rate = float(np.clip(rate, 1e-6, 1.0 - 1e-6))
                return float(np.quantile(probs_v, 1.0 - rate))

            def threshold_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
                if args.threshold_objective == "f1_pos":
                    return float(f1_score(y_true, y_pred, zero_division=0))
                if args.threshold_objective == "f1_macro":
                    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
                if args.threshold_objective == "balanced_accuracy":
                    return float(balanced_accuracy_score(y_true, y_pred))
                return float(matthews_corrcoef(y_true, y_pred))

            band_lo = max(0.15, y_rate - 0.35)
            band_hi = min(0.85, y_rate + 0.35)
            mixed_rows: list[tuple[float, float, float]] = []
            found_mixed = False
            for threshold in cand_ts:
                y_pred = (probs_v > threshold).astype(int)
                pred_rate = float(np.mean(y_pred))
                if pred_rate <= 0.0 or pred_rate >= 1.0:
                    continue
                found_mixed = True
                score = threshold_score(y_v, y_pred)
                balance = -abs(pred_rate - y_rate)
                mixed_rows.append((float(threshold), float(score), float(balance)))

            best_t = float(args.eval_threshold)
            best_score = float("-inf")
            search_mode = f"{args.threshold_objective}_mixed"
            if found_mixed:
                in_band = [
                    row
                    for row in mixed_rows
                    if band_lo <= float(np.mean(probs_v > row[0])) <= band_hi
                ]
                candidates = in_band if in_band else mixed_rows
                if in_band:
                    search_mode = f"{args.threshold_objective}_mixed_band"
                best_t, best_score, _ = max(candidates, key=lambda row: (row[1], row[2]))
                if args.threshold_drift_adjust:
                    train_cand = val_probs_for_threshold_search(
                        best_model,
                        train_frames,
                        feats_lstm,
                        args.window,
                        DEVICE,
                        use_interval_embedding=use_interval_embedding,
                        inference_batch_size=args.batch,
                    )
                    if train_cand is not None:
                        probs_tr, _ = train_cand
                        mean_train = float(np.mean(probs_tr))
                        mean_val = float(np.mean(probs_v))
                        drift = mean_val - mean_train
                        if abs(drift) >= 0.005:
                            threshold_adj = float(np.clip(best_t + drift, 1e-9, 1.0 - 1e-9))
                            y_adj = (probs_v > threshold_adj).astype(int)
                            pred_rate_adj = float(np.mean(y_adj))
                            if 0.0 < pred_rate_adj < 1.0:
                                score_adj = threshold_score(y_v, y_adj)
                                if score_adj >= best_score - 0.05:
                                    best_t = threshold_adj
                                    best_score = float(score_adj)
                                    search_mode = f"{search_mode}_drift"

            if not found_mixed:
                best_t = prevalence_threshold(y_rate)
                y_pred = (probs_v > best_t).astype(int)
                pred_rate_fb = float(np.mean(y_pred))
                if pred_rate_fb <= 0.0 or pred_rate_fb >= 1.0:
                    best_t = float(args.eval_threshold)
                    y_pred = (probs_v > best_t).astype(int)
                    search_mode = "eval_threshold_fallback"
                    print(
                        "[WARN] Val probs ~ constant; cannot split by threshold. "
                        f"Using --eval_threshold={best_t:.4f}."
                    )
                else:
                    search_mode = "prevalence_quantile_fallback"
                    print(
                        "[WARN] No mixed predictions on coarse grid; using prevalence-matched "
                        f"quantile t={best_t:.4f} (val pos rate={y_rate:.4f})."
                    )
                best_score = threshold_score(y_v, y_pred)

            effective_thr = best_t
            threshold_search_info = {
                "best_threshold": best_t,
                "best_val_objective": best_score,
                "objective": args.threshold_objective,
                "search_mode": search_mode,
                "prob_min": lo,
                "prob_max": hi,
                "mixed_band": {
                    "pred_pos_rate_lo": band_lo,
                    "pred_pos_rate_hi": band_hi,
                },
                "candidates_note": "uniform + percentiles + linspace(min_prob,max_prob); skip all-0/all-1",
            }
            print(
                f"[INFO] Val threshold ({search_mode}): {effective_thr:.4f} "
                f"({args.threshold_objective}={best_score:.4f}; "
                f"base --eval_threshold was {args.eval_threshold:.4f})"
            )
            pred_rate_at_thr = float(np.mean(probs_v > effective_thr))
            if pred_rate_at_thr > 0.85:
                print(
                    f"[WARN] Chosen threshold ({effective_thr:.4f}) predicts UP "
                    f"{pred_rate_at_thr:.1%} of the time. Model outputs appear nearly constant."
                )
            elif pred_rate_at_thr < 0.15:
                print(
                    f"[WARN] Chosen threshold ({effective_thr:.4f}) predicts DOWN "
                    f"{1 - pred_rate_at_thr:.1%} of the time. Model outputs appear nearly constant."
                )
        else:
            print("[WARN] Val set empty for threshold search; using --eval_threshold for metrics.")
    else:
        print(f"[INFO] Fixed eval threshold (no search): {effective_thr:.4f}")

    val_metrics = evaluate_split(
        best_model,
        val_frames,
        feats_lstm,
        args.window,
        DEVICE,
        effective_thr,
        use_interval_embedding=use_interval_embedding,
        inference_batch_size=args.batch,
    )
    test_metrics = evaluate_split(
        best_model,
        test_frames,
        feats_lstm,
        args.window,
        DEVICE,
        effective_thr,
        use_interval_embedding=use_interval_embedding,
        inference_batch_size=args.batch,
    )
    wf_metrics = walk_forward_metrics(
        best_model,
        test_frames,
        feats_lstm,
        args.window,
        DEVICE,
        effective_thr,
        args.walk_forward_folds,
        use_interval_embedding=use_interval_embedding,
        inference_batch_size=args.batch,
    )

    print_eval_section(val_metrics, test_metrics, args.window)
    for split_name, split_metrics in [("VAL", val_metrics), ("TEST", test_metrics)]:
        if split_metrics is None:
            continue
        pred_positive_rate = split_metrics.get("pred_positive_rate", float("nan"))
        if pred_positive_rate > 0.85:
            print(
                f"[WARN] {split_name}: P(pred=1)={pred_positive_rate:.1%} "
                "model is nearly always predicting UP."
            )
        elif pred_positive_rate < 0.15:
            print(
                f"[WARN] {split_name}: P(pred=1)={pred_positive_rate:.1%} "
                "model is nearly always predicting DOWN."
            )
    if wf_metrics:
        print(f"[WF] collected {len(wf_metrics)} fold metrics")

    meta = {
        "seed": args.seed,
        "window": args.window,
        "eval_threshold": effective_thr,
        "eval_threshold_base": args.eval_threshold,
        "threshold_objective": args.threshold_objective,
        "threshold_drift_adjust": args.threshold_drift_adjust,
        "val_threshold_search": args.val_threshold_search,
        "threshold_search": threshold_search_info,
        "num_layers": args.num_layers,
        "lr_schedule_patience": args.lr_schedule_patience,
        "early_stop_metric": args.early_stop_metric,
        "no_pos_weight": args.no_pos_weight,
        "pos_weight_min": args.pos_weight_min,
        "label_threshold": args.label_threshold,
        "label_threshold_quantile": args.label_threshold_quantile,
        "label_pos_ratio": pos_ratio,
        "balanced_pos_weight": balanced_pw,
        "walk_forward_folds": args.walk_forward_folds,
        "features": feats,
        "feats_lstm": feats_lstm,
        "use_interval_embedding": use_interval_embedding,
        "num_interval_embeddings": NUM_INTERVAL_EMBEDDINGS if use_interval_embedding else None,
        "interval_embed_dim": INTERVAL_EMBED_DIM if use_interval_embedding else None,
        "features_scaled": feats_scaled,
        "interval_id_map": {
            **{str(i): name for i, name in enumerate(INTERVAL_ID_ORDER)},
            str(INTERVAL_ID_UNKNOWN): "unknown",
        },
        "use_attn": args.use_attn,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1 - args.train_ratio - args.val_ratio,
        "split_sizes_rows": {
            "train": int(sum(len(frame) for frame in train_frames)),
            "val": int(sum(len(frame) for frame in val_frames)),
            "test": int(sum(len(frame) for frame in test_frames)),
        },
        "split_sizes_sequences": {
            "train": int(X.shape[0]),
            "val": int(X_val.shape[0]),
            "test": int(test_seq_count),
        },
        "epoch_history": epoch_history,
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
        "data_fingerprints": {str(Path(path)): file_sha256(path) for path in sorted(csv_list)},
        "metrics": {
            "val": val_metrics,
            "test": test_metrics,
            "walk_forward": wf_metrics,
        },
        "reproducibility": {
            "command": "python " + " ".join(args.command_argv),
            "git_commit": get_git_commit_hash(),
        },
        "protocol": protocol if protocol else None,
        "protocol_path": args.protocol,
        "model_path": str(save_path),
        "scaler_path": str(scaler_path),
    }
    dump_json(meta_path, meta)
    print(f"[OK] scaler saved: {scaler_path}")
    print(f"[OK] metadata saved: {meta_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_training(args)


if __name__ == "__main__":
    raise SystemExit(main())
