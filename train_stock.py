#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_stock.py   ── 只做訓練
用法範例：
  python train_stock.py \
      --csvs AAPL_d.csv AAPL_1m.csv \
      --save_model dir_model.pt \
      --window 30 --epochs 40 --use_attn
"""
import os, glob, sys, math, argparse, json, pickle, random, hashlib, subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from app.paths import MODEL_DIR
import numpy as np
import pandas as pd
import torch
from torch import nn
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ─── 裝置 ──────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Technical Indicators ─────────────────
from ta.momentum   import RSIIndicator
from ta.trend      import MACD, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# ╭────────── Feature Engineering ──────────╮
def fe(df: pd.DataFrame):
    df = df.copy()
    df["rsi"]   = RSIIndicator(df["close"]).rsi()
    df["macd"]  = MACD(df["close"]).macd_diff()
    df["bbw"]   = BollingerBands(df["close"]).bollinger_wband()
    df["atr"]   = AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
    df["vma20"] = SMAIndicator(df["volume"], 20).sma_indicator()
    df["v_ratio"] = df["volume"] / df["vma20"]
    df["body"]  = (df["open"] - df["close"]).abs()
    df["range"] = df["high"] - df["low"]
    df["log_ret"]  = np.log(df["close"]).diff().shift(-1)
    df["direction"] = (df["log_ret"] > 0).astype(np.float32)
    return df.dropna().reset_index(drop=True)

def build_seq(frame, feats, window):
    X, y = [], []
    v = frame[feats].values.astype(np.float32)
    d = frame["direction"].values.astype(np.float32)
    for i in range(window, len(frame)):
        X.append(v[i-window:i])
        y.append(d[i])
    return np.array(X), np.array(y).reshape(-1, 1)

def build_seq_multi(frames, feats, window):
    xs, ys = [], []
    for fr in frames:
        X_part, y_part = build_seq(fr, feats, window)
        if X_part.shape[0] > 0:
            xs.append(X_part)
            ys.append(y_part)
    if not xs:
        return np.empty((0, window, len(feats)), dtype=np.float32), np.empty((0, 1), dtype=np.float32)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

def evaluate_split(model: nn.Module, frame: pd.DataFrame, feats, window: int, device, threshold: float):
    X_eval, y_eval = build_seq_multi(frame, feats, window)
    if X_eval.shape[0] == 0:
        return None
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_eval).to(device))
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    y_true = y_eval.reshape(-1).astype(int)
    y_pred = (probs > threshold).astype(int)
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }

# ╭───────────── 模型 ──────────────────────╮
class LSTMDir(nn.Module):
    def __init__(self, d_in:int, hid:int=128, att:bool=False):
        super().__init__()
        self.att  = att
        self.lstm = nn.LSTM(d_in, hid, batch_first=True)
        if att:
            self.w = nn.Linear(hid, 1, bias=False)
        self.fc = nn.Linear(hid, 1)
    def forward(self, x):
        o, _ = self.lstm(x)
        if self.att:
            a = torch.softmax(self.w(o), dim=1)
            o = (a * o).sum(1)
        else:
            o = o[:, -1]
        return self.fc(o)

# ╭───────────── CLI ───────────────────────╮
ap = argparse.ArgumentParser()
ap.add_argument("--csvs", nargs="*", help="多個 csv 檔路徑")
ap.add_argument("--csv_dir", help="含一批 csv 的資料夾")
ap.add_argument("--ticker", help="只用此代號的 csv (與 --csv_dir 合用時篩選，如 AAPL)；也決定 model/{ticker}/ 資料夾")
ap.add_argument("--save_model", required=True)
ap.add_argument("--window", type=int, default=30)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch",  type=int, default=256)
ap.add_argument("--lr",     type=float, default=1e-3)
ap.add_argument("--patience", type=int, default=6)
ap.add_argument("--use_attn", action="store_true")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--train_ratio", type=float, default=0.7)
ap.add_argument("--val_ratio", type=float, default=0.15)
ap.add_argument("--eval_threshold", type=float, default=0.5)
ap.add_argument("--walk_forward_folds", type=int, default=0, help="Optional rolling evaluation folds on test split")
ap.add_argument("--protocol", type=str, help="Path to experiment protocol json")

# ========= 互動模式補丁 (for train_stock.py) =========
if len(sys.argv) == 1:
    print("\n=== Train ‧ Interactive mode ===")
    csv_mode = input("用資料夾還是逐檔？(d=資料夾 / f=多檔) [d] ").strip().lower()
    if csv_mode.startswith("f"):
        csvs_raw = input("請輸入多個 CSV 路徑 (以空白分隔): ").strip()
        csvs = [p.strip().strip('"').strip("'") for p in csvs_raw.split()]
        sys.argv += ["--csvs", *csvs]
    else:
        csv_dir = input("請輸入資料夾路徑 (內含一批 *.csv): ").strip().strip('"').strip("'") or "."
        sys.argv += ["--csv_dir", csv_dir]

    mdl = input("模型輸出檔名 (如 dir_model.pt) [dir_model.pt] ").strip() or "dir_model.pt"
    if not mdl.endswith(".pt"):
        mdl += ".pt"
    sys.argv += ["--save_model", mdl]

    win = input("window 長度 [30] ").strip() or "30"
    epc = input("epochs [40] ").strip() or "40"
    att = input("使用 Attention? (y/n) [n] ").strip().lower().startswith("y")
    sys.argv += ["--window", win, "--epochs", epc]
    if att:
        sys.argv.append("--use_attn")
# ========= 補丁結束 ====================================

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

# ╭────────── 讀檔 & FE（含頻率判斷） ──────────╮
def read_and_fe(path: str):                           # ★ NEW
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    # 0 = 日線、1 = 分鐘線（只要時間成份非 00:00:00 視為分鐘線）
    is_min = (df["date"].dt.hour != 0) | (df["date"].dt.minute != 0) | (df["date"].dt.second != 0)
    df["granularity"] = is_min.astype(np.int8)        # ★ NEW
    df = df.sort_values("date")
    out = fe(df)
    out["series_id"] = Path(path).stem
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

def walk_forward_metrics(model, test_frames, feats, window, device, threshold, folds: int):
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
            met = evaluate_split(model, [chunk], feats, window, device, threshold)
            if met is not None:
                met["series_index"] = idx
                met["fold_index"] = fidx
                per_fold.append(met)
    return per_fold

# ╭─────────────── 主流程 ──────────────────╮
csv_list = args.csvs or []
if args.csv_dir:
    all_csv = glob.glob(os.path.join(args.csv_dir, "*.csv"))
    if args.ticker:
        ticker_upper = args.ticker.strip().upper()
        csv_list += [p for p in all_csv if Path(p).stem.upper().startswith(ticker_upper + "_")]
        if not csv_list:
            sys.exit(f"[ERROR] No CSV in {args.csv_dir!r} for ticker {ticker_upper} (e.g. {ticker_upper}_5y_1d.csv)")
    else:
        csv_list += all_csv
if not csv_list:
    sys.exit("[ERROR] Must specify --csvs or --csv_dir")

# Always save under model/{stock-symbol}/ (symbol from --ticker or first CSV)
symbol = (args.ticker.strip().upper() if args.ticker else Path(csv_list[0]).stem.split("_")[0].upper())
model_dir = MODEL_DIR / symbol
model_dir.mkdir(parents=True, exist_ok=True)
save_path = model_dir / (Path(args.save_model).name or "dir_model.pt")

frames = [read_and_fe(p) for p in csv_list]           # ★ NEW

FEATS = [c for c in frames[0].columns if c not in ["date", "log_ret", "direction", "series_id"]]
if "granularity" not in FEATS:                        # ★ NEW
    FEATS.append("granularity")

split_triplets = [split_frame_by_ratio(fr, args.train_ratio, args.val_ratio) for fr in frames]
train_frames = [t[0] for t in split_triplets if len(t[0]) > 0]
val_frames = [t[1] for t in split_triplets if len(t[1]) > 0]
test_frames = [t[2] for t in split_triplets if len(t[2]) > 0]

if not train_frames:
    sys.exit("[ERROR] Train split is empty for all series")

# Fit scaler on train only (strict leakage control)
sc = StandardScaler()
train_stack = pd.concat(train_frames, ignore_index=True)
sc.fit(train_stack[FEATS])
for fr in train_frames:
    fr[FEATS] = sc.transform(fr[FEATS]).astype(np.float32)
for fr in val_frames:
    fr[FEATS] = sc.transform(fr[FEATS]).astype(np.float32)
for fr in test_frames:
    fr[FEATS] = sc.transform(fr[FEATS]).astype(np.float32)

X, y = build_seq_multi(train_frames, FEATS, args.window)
if X.shape[0] == 0:
    sys.exit(f"[ERROR] Train sequences are insufficient for window={args.window}")
X_val, y_val = build_seq_multi(val_frames, FEATS, args.window)
if X_val.shape[0] == 0:
    sys.exit(f"[ERROR] Validation sequences are insufficient for window={args.window}")

ds = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
dl_gen = torch.Generator()
dl_gen.manual_seed(args.seed)
dl = torch.utils.data.DataLoader(ds, args.batch, shuffle=True, generator=dl_gen)

model = LSTMDir(len(FEATS), att=args.use_attn).to(DEVICE)
pos_ratio = y.mean(); neg_ratio = 1 - pos_ratio
safe_pos = max(float(pos_ratio), 1e-6)
crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg_ratio/safe_pos]).to(DEVICE))
opt  = torch.optim.Adam(model.parameters(), args.lr)
val_xb = torch.tensor(X_val).to(DEVICE)
val_yb = torch.tensor(y_val).to(DEVICE)

best, wait = math.inf, 0
for ep in range(args.epochs):
    model.train(); loss_sum = 0.0
    for xb, yb in dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(xb), yb); loss.backward(); opt.step()
        loss_sum += loss.item() * len(xb)
    avg_train = loss_sum / len(ds)
    model.eval()
    with torch.no_grad():
        avg_val = crit(model(val_xb), val_yb).item()
    print(f"[{ep+1:03d}] train_loss={avg_train:.4f} val_loss={avg_val:.4f}")
    if avg_val < best:
        best, wait = avg_val, 0
        torch.save(model.state_dict(), save_path)
    else:
        wait += 1
        if wait >= args.patience:
            print("[early stop]"); break
print(f"[OK] saved to {save_path}")

# Save scaler + metadata next to model
scaler_path = save_path.with_suffix(".scaler.pkl")
meta_path = save_path.with_suffix(".meta.json")
with open(scaler_path, "wb") as f:
    pickle.dump(sc, f)

# Evaluate best checkpoint on val/test splits
best_model = LSTMDir(len(FEATS), att=args.use_attn).to(DEVICE)
best_model.load_state_dict(torch.load(save_path, map_location=DEVICE))
val_metrics = evaluate_split(best_model, val_frames, FEATS, args.window, DEVICE, args.eval_threshold)
test_metrics = evaluate_split(best_model, test_frames, FEATS, args.window, DEVICE, args.eval_threshold)
wf_metrics = walk_forward_metrics(
    best_model, test_frames, FEATS, args.window, DEVICE, args.eval_threshold, args.walk_forward_folds
)

if val_metrics is None:
    print(f"[WARN] val rows are insufficient for window={args.window}, skip val metrics")
else:
    print(
        f"[VAL] n={val_metrics['n_samples']} "
        f"acc={val_metrics['accuracy']:.4f} "
        f"prec={val_metrics['precision']:.4f} "
        f"rec={val_metrics['recall']:.4f} "
        f"f1={val_metrics['f1']:.4f} "
        f"thr={val_metrics['threshold']:.2f}"
    )
if test_metrics is None:
    print(f"[WARN] test rows are insufficient for window={args.window}, skip test metrics")
else:
    print(
        f"[TEST] n={test_metrics['n_samples']} "
        f"acc={test_metrics['accuracy']:.4f} "
        f"prec={test_metrics['precision']:.4f} "
        f"rec={test_metrics['recall']:.4f} "
        f"f1={test_metrics['f1']:.4f} "
        f"thr={test_metrics['threshold']:.2f}"
    )
if wf_metrics:
    print(f"[WF] collected {len(wf_metrics)} fold metrics")

meta = {
    "seed": args.seed,
    "window": args.window,
    "eval_threshold": args.eval_threshold,
    "walk_forward_folds": args.walk_forward_folds,
    "features": FEATS,
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
        "test": int(build_seq_multi(test_frames, FEATS, args.window)[0].shape[0]),
    },
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
print(f"[OK] saved scaler to {scaler_path}")
print(f"[OK] saved metadata to {meta_path}")
