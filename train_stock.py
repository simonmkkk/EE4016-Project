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
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Feature engineering ---
def fe(df: pd.DataFrame):
    out = add_technical_indicators(df)
    out["log_ret"] = np.log(out["close"]).diff().shift(-1)
    out["direction"] = (out["log_ret"] > 0).astype(np.float32)
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
    default=0.4,
    help="Base decision threshold; also grid fallback when val search is disabled",
)
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
    default=3,
    help="Number of LSTM layers (residual between layer 2..N)",
)
ap.add_argument(
    "--lr_schedule_patience",
    type=int,
    default=5,
    help="ReduceLROnPlateau patience (epochs without val loss improvement)",
)
ap.add_argument("--walk_forward_folds", type=int, default=0, help="Optional rolling evaluation folds on test split")
ap.add_argument("--protocol", type=str, help="Path to experiment protocol json")

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
def read_and_fe(path: str):
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    # 0 = daily bar, 1 = intraday if time component is not 00:00:00
    is_min = (df["date"].dt.hour != 0) | (df["date"].dt.minute != 0) | (df["date"].dt.second != 0)
    df["granularity"] = is_min.astype(np.int8)
    df = df.sort_values("date")
    out = fe(df)
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

    if val_metrics is None:
        print(f"  {'VAL':<6}{'N/A':>8}{'-':>10}{'-':>10}{'-':>10}{'-':>10}{'-':>8}")
        print(f"  [WARN] val rows are insufficient for window={window}")
    else:
        _print_row("VAL", val_metrics)

    if test_metrics is None:
        print(f"  {'TEST':<6}{'N/A':>8}{'-':>10}{'-':>10}{'-':>10}{'-':>10}{'-':>8}")
        print(f"  [WARN] test rows are insufficient for window={window}")
    else:
        _print_row("TEST", test_metrics)

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
            ("lr_sched_patience", str(args.lr_schedule_patience)),
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
    fr = read_and_fe(p)
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

FEATS = [c for c in frames[0].columns if c not in ["date", "log_ret", "direction", "series_id"]]
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
print_kv_section(
    "TRAINING DATA SUMMARY",
    [
        ("features", str(len(FEATS))),
        ("lstm_inputs", str(len(FEATS_LSTM))),
        ("interval_embedding", str(USE_INTERVAL_EMBEDDING)),
        ("train rows", str(sum(len(fr) for fr in train_frames))),
        ("val rows", str(sum(len(fr) for fr in val_frames))),
        ("test rows", str(sum(len(fr) for fr in test_frames))),
        ("train seq", str(X.shape[0])),
        ("val seq", str(X_val.shape[0])),
        ("test seq", str(test_seq_count)),
    ],
    border="-",
)

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
).to(DEVICE)
pos_ratio = y.mean(); neg_ratio = 1 - pos_ratio
safe_pos = max(float(pos_ratio), 1e-6)
crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg_ratio/safe_pos]).to(DEVICE))
opt = torch.optim.Adam(model.parameters(), args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode="min", factor=0.5, patience=args.lr_schedule_patience
)
val_xb = torch.tensor(X_val).to(DEVICE)
val_yb = torch.tensor(y_val).to(DEVICE)
val_ivb = torch.tensor(iv_val).to(DEVICE) if iv_val is not None else None

best, wait = math.inf, 0
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
            avg_val = crit(model(val_xb, val_ivb), val_yb).item()
        else:
            avg_val = crit(model(val_xb), val_yb).item()
    improved = avg_val < best
    if improved:
        best, wait = avg_val, 0
        torch.save(model.state_dict(), save_path)
    else:
        wait += 1

    scheduler.step(avg_val)

    epoch_history.append(
        {
            "epoch": ep + 1,
            "epochs": args.epochs,
            "train_loss": float(avg_train),
            "val_loss": float(avg_val),
            "best_val_loss": float(best),
            "wait": int(wait),
            "patience": int(args.patience),
            "improved": bool(improved),
        }
    )
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
    )
    if cand is not None:
        probs_v, y_v = cand
        best_t, best_f1 = float(args.eval_threshold), -1.0
        for t in np.arange(0.30, 0.62, 0.02):
            f1 = f1_score(y_v, (probs_v > t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        effective_thr = best_t
        threshold_search_info = {"best_threshold": best_t, "best_val_f1": best_f1, "grid": "0.30:0.02:0.60"}
        print(
            f"[INFO] Val F1-optimal threshold: {effective_thr:.4f} "
            f"(F1={best_f1:.4f}; base --eval_threshold was {args.eval_threshold:.4f})"
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
)
test_metrics = evaluate_split(
    best_model,
    test_frames,
    FEATS_LSTM,
    args.window,
    DEVICE,
    effective_thr,
    use_interval_embedding=USE_INTERVAL_EMBEDDING,
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
)

print_eval_section(val_metrics, test_metrics, args.window)
if wf_metrics:
    print(f"[WF] collected {len(wf_metrics)} fold metrics")

meta = {
    "seed": args.seed,
    "window": args.window,
    "eval_threshold": effective_thr,
    "eval_threshold_base": args.eval_threshold,
    "val_threshold_search": args.val_threshold_search,
    "threshold_search": threshold_search_info,
    "num_layers": args.num_layers,
    "lr_schedule_patience": args.lr_schedule_patience,
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
