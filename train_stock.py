#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_stock.py -- training only.

Example:
  python train_stock.py \\
      --csvs historical_data/AAPL_1d_10y.csv \\
      --ticker AAPL --save_model model.pt \\
      --window 30 --epochs 40 --use_attn \\
      --eval_threshold 0.4
"""
import os, glob, sys, math, argparse, json, pickle, random, hashlib, subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from app.paths import MODEL_DIR
from app.fe import add_technical_indicators
from app.constants import INTERVAL_ID_ORDER, INTERVAL_ID_UNKNOWN, interval_id_from_csv_stem
from app.model import LSTMDir, lstm_feature_columns
from app.sequences import build_seq_multi, evaluate_split, val_probs_for_threshold_search
import numpy as np
import pandas as pd
import torch
from torch import nn
import sklearn
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Feature engineering ---
def fe(df: pd.DataFrame, label_threshold: float = 0.0, label_threshold_quantile: float | None = None):
    out = add_technical_indicators(df)
    out["log_ret"] = np.log(out["close"]).diff().shift(-1)
    if label_threshold_quantile is not None:
        # Per-series adaptive threshold: UP = top (1 - quantile) fraction of log returns.
        # Computed on the full series before train/val/test split (it's a label definition, not a feature).
        actual_thr = float(np.nanquantile(out["log_ret"].dropna().values, label_threshold_quantile))
    else:
        actual_thr = label_threshold
    out["direction"] = (out["log_ret"] > actual_thr).astype(np.float32)
    out["_label_thr_used"] = np.float32(actual_thr)
    return out.dropna().reset_index(drop=True)

# --- CLI ---
ap = argparse.ArgumentParser()
ap.add_argument("--csvs", nargs="*", help="paths to one or more CSV files")
ap.add_argument("--csv_dir", help="directory containing CSV files to load")
ap.add_argument(
    "--ticker",
    help="filter CSVs by ticker when using --csv_dir (e.g. AAPL); also sets model/{ticker}/ output folder",
)
ap.add_argument("--save_model", required=True)
ap.add_argument("--window", type=int, default=30)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch",  type=int, default=256)
ap.add_argument("--lr",     type=float, default=1e-3)
ap.add_argument("--patience", type=int, default=15)
ap.add_argument("--use_attn", action="store_true")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--train_ratio", type=float, default=0.7)
ap.add_argument("--val_ratio", type=float, default=0.15)
ap.add_argument(
    "--eval_threshold",
    type=float,
    default=0.5,
    help="Base decision threshold; also grid fallback when val search is disabled",
)
ap.add_argument(
    "--threshold_objective",
    type=str,
    default="f1_macro",
    choices=["f1_pos", "f1_macro", "balanced_accuracy", "mcc"],
    help=(
        "Objective used by val threshold search. "
        "f1_macro is safer than f1_pos on weak-signal data because it penalizes one-sided predictions."
    ),
)
thr_drift_grp = ap.add_mutually_exclusive_group()
thr_drift_grp.add_argument(
    "--threshold-drift-adjust",
    dest="threshold_drift_adjust",
    action="store_true",
    help=(
        "Adjust selected val threshold by (val_prob_mean - train_prob_mean) to mitigate train/val temporal drift "
        "before evaluating val/test. Enabled by default."
    ),
)
thr_drift_grp.add_argument(
    "--no-threshold-drift-adjust",
    dest="threshold_drift_adjust",
    action="store_false",
    help="Disable train->val probability drift adjustment for threshold.",
)
ap.set_defaults(threshold_drift_adjust=True)
thr_grp = ap.add_mutually_exclusive_group()
thr_grp.add_argument(
    "--val-threshold-search",
    dest="val_threshold_search",
    action="store_true",
    help="Pick threshold on val to maximize F1 (default)",
)
thr_grp.add_argument(
    "--no-val-threshold-search",
    dest="val_threshold_search",
    action="store_false",
    help="Use fixed --eval_threshold for val/test metrics",
)
ap.set_defaults(val_threshold_search=True)
ap.add_argument(
    "--num_layers",
    type=int,
    default=2,
    help="Number of LSTM layers (residual between layer 2..N)",
)
ap.add_argument(
    "--lr_schedule_patience",
    type=int,
    default=5,
    help="ReduceLROnPlateau patience (epochs without val loss improvement)",
)
ap.add_argument("--walk_forward_folds", type=int, default=0, help="Optional rolling evaluation folds on test split")
ap.add_argument(
    "--early_stop_metric",
    type=str,
    default="val_loss",
    choices=["val_loss", "val_f1"],
    help="Early stopping monitor: val_loss (lower=better, default) or val_f1 (higher=better, threshold-sensitive)",
)
ap.add_argument("--protocol", type=str, help="Path to experiment protocol json")
ap.add_argument(
    "--no-pos-weight",
    "--no_pos_weight",
    action="store_true",
    help="Unweighted BCEWithLogitsLoss (diagnostic; ignores neg/pos balance)",
)
ap.add_argument(
    "--pos-weight-min",
    type=float,
    default=None,
    metavar="W",
    help="Clamp balanced pos_weight=max(neg/pos, W), e.g. 0.7. Ignored with --no-pos-weight.",
)
ap.add_argument(
    "--dropout",
    type=float,
    default=0.4,
    help="Dropout probability applied to the LSTM pooled vector before the FC layer (default: 0.4).",
)
ap.add_argument(
    "--weight_decay",
    type=float,
    default=1e-4,
    help="L2 weight decay for Adam optimiser (default: 1e-4). Helps prevent early overfitting.",
)
ap.add_argument(
    "--label_threshold",
    type=float,
    default=0.0,
    metavar="T",
    help=(
        "Minimum log-return to label a bar as UP (default: 0.0 = any positive return). "
        "E.g. 0.003 means only bars with log_ret > 0.3%% are labelled UP. "
        "Ignored when --label_threshold_quantile is set."
    ),
)
ap.add_argument(
    "--label_threshold_quantile",
    type=float,
    default=None,
    metavar="Q",
    help=(
        "Per-series adaptive label threshold (default: None = use --label_threshold). "
        "When set, the threshold for each CSV is computed as quantile(log_ret, Q) of that series. "
        "E.g. 0.55 means the top 45%% of log-returns are labelled UP, keeping class balance "
        "consistent across different bar sizes (1m, 1h, 1d, etc.). "
        "Recommended range: 0.50-0.65."
    ),
)

# --- Interactive mode when no CLI args ---
if len(sys.argv) == 1:
    print("\n=== Train -- interactive mode ===")
    csv_mode = input("Load from directory or list files? (d=directory / f=files) [d] ").strip().lower()
    if csv_mode.startswith("f"):
        csvs_raw = input("Enter CSV paths (space-separated): ").strip()
        csvs = [p.strip().strip('"').strip("'") for p in csvs_raw.split()]
        sys.argv += ["--csvs", *csvs]
    else:
        csv_dir = input("Directory path (contains *.csv): ").strip().strip('"').strip("'") or "."
        sys.argv += ["--csv_dir", csv_dir]

    mdl = input("Output model filename (e.g. model.pt) [model.pt] ").strip() or "model.pt"
    if not mdl.endswith(".pt"):
        mdl += ".pt"
    sys.argv += ["--save_model", mdl]

    win = input("window length [30] ").strip() or "30"
    epc = input("epochs [40] ").strip() or "40"
    att = input("Use attention? (y/n) [n] ").strip().lower().startswith("y")
    sys.argv += ["--window", win, "--epochs", epc]
    if att:
        sys.argv.append("--use_attn")
# --- end interactive patch ---

args = ap.parse_args()

protocol = {}
if args.protocol:
    try:
        with open(args.protocol, "r", encoding="utf-8") as f:
            protocol = json.load(f)
    except Exception as e:
        sys.exit(f"[ERROR] Failed to read protocol {args.protocol}: {e}")
    split_cfg = protocol.get("split", {})
    if split_cfg.get("method") == "ratio":
        args.train_ratio = float(split_cfg.get("train_ratio", args.train_ratio))
        args.val_ratio = float(split_cfg.get("val_ratio", args.val_ratio))

if not (0 < args.train_ratio < 1 and 0 < args.val_ratio < 1 and args.train_ratio + args.val_ratio < 1):
    sys.exit("[ERROR] train_ratio and val_ratio must be in (0,1), and train_ratio + val_ratio < 1")
if not (0 < args.eval_threshold < 1):
    sys.exit("[ERROR] eval_threshold must be in (0,1)")
if args.num_layers < 1:
    sys.exit("[ERROR] num_layers must be >= 1")
if args.walk_forward_folds < 0:
    sys.exit("[ERROR] walk_forward_folds must be >= 0")
if args.pos_weight_min is not None and args.pos_weight_min <= 0:
    sys.exit("[ERROR] --pos-weight-min must be > 0")
if args.no_pos_weight and args.pos_weight_min is not None:
    print("[WARN] --no-pos-weight set; --pos-weight-min ignored")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)

# --- Load CSV + feature engineering (bar frequency hint) ---
def read_and_fe(path: str, label_threshold: float = 0.0, label_threshold_quantile: float | None = None):
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    # 0 = daily bar, 1 = intraday if time component is not 00:00:00
    is_min = (df["date"].dt.hour != 0) | (df["date"].dt.minute != 0) | (df["date"].dt.second != 0)
    df["granularity"] = is_min.astype(np.int8)
    df = df.sort_values("date")
    out = fe(df, label_threshold=label_threshold, label_threshold_quantile=label_threshold_quantile)
    out["series_id"] = Path(path).stem
    # Ordinal interval id from filename (0..len-1); unknown token -> INTERVAL_ID_UNKNOWN (excluded from StandardScaler).
    out["interval_id"] = np.float32(interval_id_from_csv_stem(Path(path).stem))
    return out

def split_frame_by_ratio(frame: pd.DataFrame, train_ratio: float, val_ratio: float):
    n = len(frame)
    tr_end = int(n * train_ratio)
    va_end = int(n * (train_ratio + val_ratio))
    train = frame.iloc[:tr_end].copy()
    val = frame.iloc[tr_end:va_end].copy()
    test = frame.iloc[va_end:].copy()
    return train, val, test

def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def get_git_commit_hash() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT)
        return out.decode("utf-8").strip()
    except Exception:
        return None

def walk_forward_metrics(
    model,
    test_frames,
    feats_lstm: list[str],
    window,
    device,
    threshold,
    folds: int,
    *,
    use_interval_embedding: bool,
    inference_batch_size: int = 256,
):
    if folds <= 1:
        return []
    per_fold = []
    for idx, fr in enumerate(test_frames):
        n = len(fr)
        if n <= window + folds:
            continue
        step = n // folds
        for fidx in range(folds):
            start = fidx * step
            end = n if fidx == folds - 1 else (fidx + 1) * step
            chunk = fr.iloc[start:end].copy()
            met = evaluate_split(
                model,
                [chunk],
                feats_lstm,
                window,
                device,
                threshold,
                use_interval_embedding=use_interval_embedding,
                inference_batch_size=inference_batch_size,
            )
            if met is not None:
                met["series_index"] = idx
                met["fold_index"] = fidx
                per_fold.append(met)
    return per_fold


LINE_WIDTH = 70


def print_kv_section(title: str, rows: list[tuple[str, str]], border: str = "="):
    print("\n" + border * LINE_WIDTH)
    print(title)
    print(border * LINE_WIDTH)
    for key, value in rows:
        print(f"  {key:<14}: {value}")
    print(border * LINE_WIDTH)


def print_eval_section(val_metrics, test_metrics, window: int):
    print("\n" + "=" * LINE_WIDTH)
    print("FINAL EVALUATION")
    print("=" * LINE_WIDTH)

    if val_metrics is None and test_metrics is None:
        print(f"[WARN] val/test rows are insufficient for window={window}")
        return

    print(f"  {'split':<6}{'n':>8}{'acc':>10}{'prec':>10}{'rec':>10}{'f1':>10}{'thr':>8}")
    print("  " + "-" * (LINE_WIDTH - 2))

    def _print_row(name: str, metrics):
        print(
            f"  {name:<6}"
            f"{metrics['n_samples']:>8}"
            f"{metrics['accuracy']:>10.4f}"
            f"{metrics['precision']:>10.4f}"
            f"{metrics['recall']:>10.4f}"
            f"{metrics['f1']:>10.4f}"
            f"{metrics['threshold']:>8.2f}"
        )

    def _print_conf(name: str, metrics):
        if metrics is None or "tn" not in metrics:
            return
        m = metrics
        print(
            f"  {name:<6}"
            f"  TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}  "
            f"P(pred=1)={m['pred_positive_rate']:.4f}"
        )

    if val_metrics is None:
        print(f"  {'VAL':<6}{'N/A':>8}{'-':>10}{'-':>10}{'-':>10}{'-':>10}{'-':>8}")
        print(f"  [WARN] val rows are insufficient for window={window}")
    else:
        _print_row("VAL", val_metrics)
        _print_conf("VAL", val_metrics)

    if test_metrics is None:
        print(f"  {'TEST':<6}{'N/A':>8}{'-':>10}{'-':>10}{'-':>10}{'-':>10}{'-':>8}")
        print(f"  [WARN] test rows are insufficient for window={window}")
    else:
        _print_row("TEST", test_metrics)
        _print_conf("TEST", test_metrics)

    print("=" * LINE_WIDTH)


def print_training_header(symbol, csv_list, save_path, args):
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
            ("pos_weight_min", str(args.pos_weight_min) if args.pos_weight_min is not None else "-"),
            ("dropout", f"{args.dropout:.2f}"),
            ("weight_decay", f"{args.weight_decay:.2e}"),
            ("label_threshold",
             f"quantile={args.label_threshold_quantile:.2f} (per-series adaptive)"
             if args.label_threshold_quantile is not None
             else (f"{args.label_threshold:.4f} (any positive)" if args.label_threshold == 0.0
                   else f"{args.label_threshold:.4f} (log_ret > {args.label_threshold:.4f})")),
            ("split ratios", f"train={args.train_ratio:.2f} val={args.val_ratio:.2f} test={test_ratio:.2f}"),
        ],
        border="=",
    )
    print("  input csvs")
    for path in csv_list:
        print(f"    - {path}")
    print("-" * LINE_WIDTH)

# --- Main training flow ---
csv_list = args.csvs or []
if args.csv_dir:
    all_csv = glob.glob(os.path.join(args.csv_dir, "*.csv"))
    if args.ticker:
        ticker_upper = args.ticker.strip().upper()
        csv_list += [p for p in all_csv if Path(p).stem.upper().startswith(ticker_upper + "_")]
        if not csv_list:
            sys.exit(f"[ERROR] No CSV in {args.csv_dir!r} for ticker {ticker_upper} (e.g. {ticker_upper}_1d_10y.csv)")
    else:
        csv_list += all_csv
if not csv_list:
    sys.exit("[ERROR] Must specify --csvs or --csv_dir")

# Always save under model/{stock-symbol}/ (symbol from --ticker or first CSV)
symbol = (args.ticker.strip().upper() if args.ticker else Path(csv_list[0]).stem.split("_")[0].upper())
model_dir = MODEL_DIR / symbol
model_dir.mkdir(parents=True, exist_ok=True)
save_path = model_dir / (Path(args.save_model).name or "model.pt")
print_training_header(symbol, csv_list, save_path, args)

empty_paths: list[str] = []
frames: list[pd.DataFrame] = []
for p in csv_list:
    fr = read_and_fe(p, label_threshold=args.label_threshold, label_threshold_quantile=args.label_threshold_quantile)
    if len(fr) == 0:
        empty_paths.append(p)
    else:
        frames.append(fr)
if empty_paths:
    print(
        f"[WARN] Skipped {len(empty_paths)} CSV(s) with no rows after feature engineering "
        "(series too short for indicators + next-bar target):"
    )
    for ep in empty_paths:
        print(f"    - {ep}")
if not frames:
    sys.exit(
        "[ERROR] No usable rows after feature engineering. "
        "Use longer histories or fewer intraday intervals with almost no bars."
    )

FEATS = [c for c in frames[0].columns if c not in ["date", "log_ret", "direction", "series_id", "_label_thr_used"]]
if "granularity" not in FEATS:
    FEATS.append("granularity")
if "interval_id" not in FEATS:
    FEATS.append("interval_id")

# interval_id is ordinal (0..K); do not pass through StandardScaler; embedding consumes raw ids in the model.
FEATS_SCALED = [c for c in FEATS if c != "interval_id"]
USE_INTERVAL_EMBEDDING = "interval_id" in FEATS
FEATS_LSTM, _ = lstm_feature_columns(FEATS, USE_INTERVAL_EMBEDDING)
INTERVAL_EMBED_DIM = 8
NUM_INTERVAL_EMBEDDINGS = INTERVAL_ID_UNKNOWN + 1

split_triplets = [split_frame_by_ratio(fr, args.train_ratio, args.val_ratio) for fr in frames]
train_frames = [t[0] for t in split_triplets if len(t[0]) > 0]
val_frames = [t[1] for t in split_triplets if len(t[1]) > 0]
test_frames = [t[2] for t in split_triplets if len(t[2]) > 0]

if not train_frames:
    sys.exit("[ERROR] Train split is empty for all series")

# Fit scaler on train only (strict leakage control)
sc = StandardScaler()
train_stack = pd.concat(train_frames, ignore_index=True)
sc.fit(train_stack[FEATS_SCALED])
for fr in train_frames:
    fr[FEATS_SCALED] = sc.transform(fr[FEATS_SCALED]).astype(np.float32)
for fr in val_frames:
    fr[FEATS_SCALED] = sc.transform(fr[FEATS_SCALED]).astype(np.float32)
for fr in test_frames:
    fr[FEATS_SCALED] = sc.transform(fr[FEATS_SCALED]).astype(np.float32)

X, y, iv_train = build_seq_multi(
    train_frames, FEATS_LSTM, args.window, use_interval_embedding=USE_INTERVAL_EMBEDDING
)
if X.shape[0] == 0:
    sys.exit(f"[ERROR] Train sequences are insufficient for window={args.window}")
X_val, y_val, iv_val = build_seq_multi(
    val_frames, FEATS_LSTM, args.window, use_interval_embedding=USE_INTERVAL_EMBEDDING
)
if X_val.shape[0] == 0:
    sys.exit(f"[ERROR] Validation sequences are insufficient for window={args.window}")

test_seq_count = build_seq_multi(
    test_frames, FEATS_LSTM, args.window, use_interval_embedding=USE_INTERVAL_EMBEDDING
)[0].shape[0]
_train_all_labels = np.concatenate([fr["direction"].values for fr in train_frames])
_train_pos_ratio = float(np.mean(_train_all_labels))
if args.label_threshold_quantile is not None:
    _label_thr_note = (
        f"per-series quantile {args.label_threshold_quantile:.2f} "
        f"(top {1-args.label_threshold_quantile:.0%} UP per series; overall {_train_pos_ratio:.1%} UP in train)"
    )
elif args.label_threshold > 0.0:
    _label_thr_note = f"log_ret > {args.label_threshold:.4f} ({_train_pos_ratio:.1%} UP in train)"
else:
    _label_thr_note = f"log_ret > 0 ({_train_pos_ratio:.1%} UP in train)"
print_kv_section(
    "TRAINING DATA SUMMARY",
    [
        ("features", str(len(FEATS))),
        ("lstm_inputs", str(len(FEATS_LSTM))),
        ("interval_embedding", str(USE_INTERVAL_EMBEDDING)),
        ("label rule", _label_thr_note),
        ("train rows", str(sum(len(fr) for fr in train_frames))),
        ("val rows", str(sum(len(fr) for fr in val_frames))),
        ("test rows", str(sum(len(fr) for fr in test_frames))),
        ("train seq", str(X.shape[0])),
        ("val seq", str(X_val.shape[0])),
        ("test seq", str(test_seq_count)),
    ],
    border="-",
)
if args.label_threshold_quantile is not None:
    print("  per-series label thresholds (quantile mode):")
    for fr in frames:
        _sid = fr["series_id"].iloc[0] if "series_id" in fr.columns else "?"
        _thr = float(fr["_label_thr_used"].iloc[0]) if "_label_thr_used" in fr.columns else float("nan")
        _pup = float(fr["direction"].mean())
        print(f"    {_sid:<28}  thr={_thr:+.6f}  UP={_pup:.1%}")

if iv_train is not None:
    ds = torch.utils.data.TensorDataset(
        torch.tensor(X), torch.tensor(iv_train), torch.tensor(y)
    )
else:
    ds = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
dl_gen = torch.Generator()
dl_gen.manual_seed(args.seed)
dl = torch.utils.data.DataLoader(ds, args.batch, shuffle=True, generator=dl_gen)

model = LSTMDir(
    len(FEATS_LSTM),
    att=args.use_attn,
    num_layers=args.num_layers,
    num_intervals=(NUM_INTERVAL_EMBEDDINGS if USE_INTERVAL_EMBEDDING else None),
    embed_dim=INTERVAL_EMBED_DIM,
    dropout=args.dropout,
).to(DEVICE)
pos_ratio = float(np.mean(y))
neg_ratio = float(1.0 - pos_ratio)
safe_pos = max(pos_ratio, 1e-6)
balanced_pw = neg_ratio / safe_pos
if args.no_pos_weight:
    crit = nn.BCEWithLogitsLoss()
    print(
        f"[INFO] pos_ratio={pos_ratio:.4f} neg_ratio={neg_ratio:.4f} "
        f"balanced_pos_weight={balanced_pw:.4f} effective=unweighted (--no-pos-weight)"
    )
else:
    pos_weight_val = balanced_pw
    if args.pos_weight_min is not None:
        pos_weight_val = max(pos_weight_val, float(args.pos_weight_min))
    crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32, device=DEVICE)
    )
    clamp_note = ""
    if args.pos_weight_min is not None and pos_weight_val > balanced_pw + 1e-12:
        clamp_note = f" (clamped from {balanced_pw:.4f} by --pos-weight-min)"
    print(
        f"[INFO] pos_ratio={pos_ratio:.4f} neg_ratio={neg_ratio:.4f} "
        f"balanced_pos_weight={balanced_pw:.4f} pos_weight={pos_weight_val:.4f}{clamp_note}"
    )
opt = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode="min", factor=0.5, patience=args.lr_schedule_patience
)
val_xb = torch.tensor(X_val).to(DEVICE)
val_yb = torch.tensor(y_val).to(DEVICE)
val_ivb = torch.tensor(iv_val).to(DEVICE) if iv_val is not None else None

best = 0.0 if args.early_stop_metric == "val_f1" else math.inf
wait = 0
epoch_history: list[dict] = []
for ep in range(args.epochs):
    model.train(); loss_sum = 0.0
    if iv_train is not None:
        for xb, ivb, yb in dl:
            xb, ivb, yb = xb.to(DEVICE), ivb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb, ivb), yb); loss.backward(); opt.step()
            loss_sum += loss.item() * len(xb)
    else:
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb); loss.backward(); opt.step()
            loss_sum += loss.item() * len(xb)
    avg_train = loss_sum / len(ds)
    model.eval()
    with torch.no_grad():
        if val_ivb is not None:
            val_logits = model(val_xb, val_ivb)
        else:
            val_logits = model(val_xb)
        avg_val = crit(val_logits, val_yb).item()

        if args.early_stop_metric == "val_f1":
            _vp = torch.sigmoid(val_logits).cpu().numpy().ravel()
            _es_val = float(
                f1_score(
                    np.asarray(y_val).ravel(),
                    (_vp > 0.5).astype(int),
                    zero_division=0,
                )
            )
            improved = _es_val > best
        else:
            _es_val = avg_val
            improved = _es_val < best
    if improved:
        best, wait = _es_val, 0
        torch.save(model.state_dict(), save_path)
    else:
        wait += 1

    scheduler.step(avg_val)

    _eh: dict = {
        "epoch": ep + 1,
        "epochs": args.epochs,
        "train_loss": float(avg_train),
        "val_loss": float(avg_val),
        "early_stop_metric": args.early_stop_metric,
        "wait": int(wait),
        "patience": int(args.patience),
        "improved": bool(improved),
    }
    if args.early_stop_metric == "val_f1":
        _eh["val_f1"] = float(_es_val)
        _eh["best_val_f1"] = float(best)
    else:
        _eh["best_val_loss"] = float(best)
    epoch_history.append(_eh)
    if args.early_stop_metric == "val_f1":
        print(
            f"[E{ep+1:03d}/{args.epochs:03d}] "
            f"train={avg_train:.4f} "
            f"val_loss={avg_val:.4f} val_f1={_es_val:.4f} best_f1={best:.4f} "
            f"wait={wait}/{args.patience}"
        )
    else:
        print(
            f"[E{ep+1:03d}/{args.epochs:03d}] "
            f"train={avg_train:.4f} "
            f"val={avg_val:.4f} "
            f"best={best:.4f} "
            f"wait={wait}/{args.patience}"
        )

    if not improved and wait >= args.patience:
        print(f"[STOP] Early stopping at epoch {ep+1} (patience={args.patience})")
        break
print(f"[OK] model saved: {save_path}")

# Save scaler + metadata next to model
scaler_path = save_path.with_suffix(".scaler.pkl")
meta_path = save_path.with_suffix(".meta.json")
with open(scaler_path, "wb") as f:
    pickle.dump(sc, f)

# Evaluate best checkpoint on val/test splits
best_model = LSTMDir(
    len(FEATS_LSTM),
    att=args.use_attn,
    num_layers=args.num_layers,
    num_intervals=(NUM_INTERVAL_EMBEDDINGS if USE_INTERVAL_EMBEDDING else None),
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
        FEATS_LSTM,
        args.window,
        DEVICE,
        use_interval_embedding=USE_INTERVAL_EMBEDDING,
        inference_batch_size=args.batch,
    )
    if cand is not None:
        probs_v, y_v = cand
        y_rate = float(np.mean(y_v))
        lo, hi = float(np.min(probs_v)), float(np.max(probs_v))
        # Uniform grid + percentiles + linspace(min,max) so candidates exist between actual probs
        thr_uniform = np.arange(0.02, 0.991, 0.02)
        thr_pct = np.percentile(probs_v, np.arange(3, 100, 2))
        _parts = [thr_uniform, thr_pct]
        if hi > lo + 1e-12:
            _parts.append(
                np.linspace(lo, hi, num=min(64, max(8, int(len(probs_v) * 2))))
            )
        cand_ts = np.unique(np.clip(np.concatenate(_parts), 1e-9, 1.0 - 1e-9))

        def _prevalence_threshold(rate: float) -> float:
            """t with ~`rate` fraction of probs > t (continuous-ish probs)."""
            rate = float(np.clip(rate, 1e-6, 1.0 - 1e-6))
            return float(np.quantile(probs_v, 1.0 - rate))

        def _threshold_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            if args.threshold_objective == "f1_pos":
                return float(f1_score(y_true, y_pred, zero_division=0))
            if args.threshold_objective == "f1_macro":
                return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            if args.threshold_objective == "balanced_accuracy":
                return float(balanced_accuracy_score(y_true, y_pred))
            return float(matthews_corrcoef(y_true, y_pred))

        # Prefer thresholds whose predicted-positive rate stays in a broad, non-degenerate band.
        # This prevents selecting thresholds that look good on pos-F1 but collapse to mostly one class.
        band_lo = max(0.15, y_rate - 0.35)
        band_hi = min(0.85, y_rate + 0.35)

        mixed_rows: list[tuple[float, float, float]] = []
        found_mixed = False
        for t in cand_ts:
            y_p = (probs_v > t).astype(int)
            pr = float(np.mean(y_p))
            if pr <= 0.0 or pr >= 1.0:
                continue
            found_mixed = True
            score = _threshold_score(y_v, y_p)
            balance = -abs(pr - y_rate)
            mixed_rows.append((float(t), float(score), float(balance)))

        best_t = float(args.eval_threshold)
        best_score = float("-inf")
        search_mode = f"{args.threshold_objective}_mixed"
        if found_mixed:
            in_band = [r for r in mixed_rows if band_lo <= float(np.mean(probs_v > r[0])) <= band_hi]
            candidates = in_band if in_band else mixed_rows
            if in_band:
                search_mode = f"{args.threshold_objective}_mixed_band"
            best_t, best_score, _ = max(candidates, key=lambda x: (x[1], x[2]))

            # Optional train->val drift correction to reduce threshold brittleness under temporal shift.
            if args.threshold_drift_adjust:
                train_cand = val_probs_for_threshold_search(
                    best_model,
                    train_frames,
                    FEATS_LSTM,
                    args.window,
                    DEVICE,
                    use_interval_embedding=USE_INTERVAL_EMBEDDING,
                    inference_batch_size=args.batch,
                )
                if train_cand is not None:
                    probs_tr, _ = train_cand
                    mean_train = float(np.mean(probs_tr))
                    mean_val = float(np.mean(probs_v))
                    drift = mean_val - mean_train
                    if abs(drift) >= 0.005:
                        t_adj = float(np.clip(best_t + drift, 1e-9, 1.0 - 1e-9))
                        y_adj = (probs_v > t_adj).astype(int)
                        pr_adj = float(np.mean(y_adj))
                        if 0.0 < pr_adj < 1.0:
                            score_adj = _threshold_score(y_v, y_adj)
                            # Keep drift-corrected threshold when objective degradation is small.
                            if score_adj >= best_score - 0.05:
                                best_t = t_adj
                                best_score = float(score_adj)
                                search_mode = f"{search_mode}_drift"

        if not found_mixed:
            # Every t in grid yields all-0 or all-1 (e.g. nearly constant logits). Match label rate.
            best_t = _prevalence_threshold(y_rate)
            y_p = (probs_v > best_t).astype(int)
            pr_fb = float(np.mean(y_p))
            if pr_fb <= 0.0 or pr_fb >= 1.0:
                # Constant probs: no t can match prevalence; use CLI default
                best_t = float(args.eval_threshold)
                y_p = (probs_v > best_t).astype(int)
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
            best_score = _threshold_score(y_v, y_p)
        effective_thr = best_t
        threshold_search_info = {
            "best_threshold": best_t,
            "best_val_objective": best_score,
            "objective": args.threshold_objective,
            "search_mode": search_mode,
            "prob_min": lo,
            "prob_max": hi,
            "mixed_band": {"pred_pos_rate_lo": band_lo, "pred_pos_rate_hi": band_hi},
            "candidates_note": "uniform + percentiles + linspace(min_prob,max_prob); skip all-0/all-1",
        }
        print(
            f"[INFO] Val threshold ({search_mode}): {effective_thr:.4f} "
            f"({args.threshold_objective}={best_score:.4f}; base --eval_threshold was {args.eval_threshold:.4f})"
        )
        _pred_rate_at_thr = float(np.mean(probs_v > effective_thr))
        if _pred_rate_at_thr > 0.85:
            print(
                f"[WARN] Chosen threshold ({effective_thr:.4f}) predicts UP {_pred_rate_at_thr:.1%} of the time. "
                "Model outputs appear nearly constant — possible causes: early overfitting, insufficient data, "
                "or low feature signal. Try: increase --dropout (e.g. 0.5), increase --weight_decay (e.g. 1e-3), "
                "reduce --num_layers, or add more CSV data."
            )
        elif _pred_rate_at_thr < 0.15:
            print(
                f"[WARN] Chosen threshold ({effective_thr:.4f}) predicts DOWN {1 - _pred_rate_at_thr:.1%} of the time. "
                "Model outputs appear nearly constant — possible causes: early overfitting, insufficient data, "
                "or low feature signal. Try: increase --dropout (e.g. 0.5), increase --weight_decay (e.g. 1e-3), "
                "reduce --num_layers, or add more CSV data."
            )
    else:
        print("[WARN] Val set empty for threshold search; using --eval_threshold for metrics.")
else:
    print(f"[INFO] Fixed eval threshold (no search): {effective_thr:.4f}")

val_metrics = evaluate_split(
    best_model,
    val_frames,
    FEATS_LSTM,
    args.window,
    DEVICE,
    effective_thr,
    use_interval_embedding=USE_INTERVAL_EMBEDDING,
    inference_batch_size=args.batch,
)
test_metrics = evaluate_split(
    best_model,
    test_frames,
    FEATS_LSTM,
    args.window,
    DEVICE,
    effective_thr,
    use_interval_embedding=USE_INTERVAL_EMBEDDING,
    inference_batch_size=args.batch,
)
wf_metrics = walk_forward_metrics(
    best_model,
    test_frames,
    FEATS_LSTM,
    args.window,
    DEVICE,
    effective_thr,
    args.walk_forward_folds,
    use_interval_embedding=USE_INTERVAL_EMBEDDING,
    inference_batch_size=args.batch,
)

print_eval_section(val_metrics, test_metrics, args.window)
for _split_name, _split_met in [("VAL", val_metrics), ("TEST", test_metrics)]:
    if _split_met is None:
        continue
    _ppr = _split_met.get("pred_positive_rate", float("nan"))
    if _ppr > 0.85:
        print(
            f"[WARN] {_split_name}: P(pred=1)={_ppr:.1%} — model is nearly always predicting UP. "
            "Consider: --dropout 0.5 --weight_decay 1e-3 --num_layers 1, or add more CSV data."
        )
    elif _ppr < 0.15:
        print(
            f"[WARN] {_split_name}: P(pred=1)={_ppr:.1%} — model is nearly always predicting DOWN. "
            "Consider: --dropout 0.5 --weight_decay 1e-3 --num_layers 1, or add more CSV data."
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
    "features": FEATS,
    "feats_lstm": FEATS_LSTM,
    "use_interval_embedding": USE_INTERVAL_EMBEDDING,
    "num_interval_embeddings": NUM_INTERVAL_EMBEDDINGS if USE_INTERVAL_EMBEDDING else None,
    "interval_embed_dim": INTERVAL_EMBED_DIM if USE_INTERVAL_EMBEDDING else None,
    "features_scaled": FEATS_SCALED,
    "interval_id_map": {str(i): name for i, name in enumerate(INTERVAL_ID_ORDER)}
    | {str(INTERVAL_ID_UNKNOWN): "unknown"},
    "use_attn": args.use_attn,
    "dropout": args.dropout,
    "weight_decay": args.weight_decay,
    "train_ratio": args.train_ratio,
    "val_ratio": args.val_ratio,
    "test_ratio": 1 - args.train_ratio - args.val_ratio,
    "split_sizes_rows": {
        "train": int(sum(len(fr) for fr in train_frames)),
        "val": int(sum(len(fr) for fr in val_frames)),
        "test": int(sum(len(fr) for fr in test_frames)),
    },
    "split_sizes_sequences": {
        "train": int(X.shape[0]),
        "val": int(X_val.shape[0]),
        "test": int(
            build_seq_multi(
                test_frames, FEATS_LSTM, args.window, use_interval_embedding=USE_INTERVAL_EMBEDDING
            )[0].shape[0]
        ),
    },
    "epoch_history": epoch_history,
    "versions": {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    },
    "data_fingerprints": {
        str(Path(p)): file_sha256(p) for p in sorted(csv_list)
    },
    "metrics": {
        "val": val_metrics,
        "test": test_metrics,
        "walk_forward": wf_metrics,
    },
    "reproducibility": {
        "command": "python " + " ".join(sys.argv),
        "git_commit": get_git_commit_hash(),
    },
    "protocol": protocol if protocol else None,
    "protocol_path": args.protocol,
    "model_path": str(save_path),
    "scaler_path": str(scaler_path),
}
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"[OK] scaler saved: {scaler_path}")
print(f"[OK] metadata saved: {meta_path}")
