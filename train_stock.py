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
import os, glob, sys, math, argparse, json, pickle, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
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

def evaluate_split(model: nn.Module, frame: pd.DataFrame, feats, window: int, device, threshold: float):
    X_eval, y_eval = build_seq(frame, feats, window)
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

# ========= 互動模式補丁 (for train_stock.py) =========
if len(sys.argv) == 1:
    print("\n=== Train ‧ Interactive mode ===")
    csv_mode = input("用資料夾還是逐檔？(d=資料夾 / f=多檔) [d] ").strip().lower()
    if csv_mode.startswith("f"):
        csvs = input("請輸入多個 CSV 路徑 (以空白分隔): ").strip().split()
        sys.argv += ["--csvs", *csvs]
    else:
        csv_dir = input("請輸入資料夾路徑 (內含一批 *.csv): ").strip() or "."
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

if not (0 < args.train_ratio < 1 and 0 < args.val_ratio < 1 and args.train_ratio + args.val_ratio < 1):
    sys.exit("[ERROR] train_ratio and val_ratio must be in (0,1), and train_ratio + val_ratio < 1")
if not (0 < args.eval_threshold < 1):
    sys.exit("[ERROR] eval_threshold must be in (0,1)")

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
    return fe(df)

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
model_dir = Path("model") / symbol
model_dir.mkdir(parents=True, exist_ok=True)
save_path = model_dir / (Path(args.save_model).name or "dir_model.pt")

frames = [read_and_fe(p) for p in csv_list]           # ★ NEW
data   = pd.concat(frames).sort_values("date").reset_index(drop=True)

FEATS = [c for c in data.columns if c not in ["date", "log_ret", "direction"]]
if "granularity" not in FEATS:                        # ★ NEW
    FEATS.append("granularity")

sc = StandardScaler()

# Chronological split to avoid look-ahead leakage
n_total = len(data)
tr_end = int(n_total * args.train_ratio)
va_end = int(n_total * (args.train_ratio + args.val_ratio))
if tr_end <= args.window or va_end <= tr_end:
    sys.exit("[ERROR] Not enough rows after split; adjust ratios or provide more data")

train_df = data.iloc[:tr_end].copy()
val_df = data.iloc[tr_end:va_end].copy()
test_df = data.iloc[va_end:].copy()
if len(test_df) == 0:
    sys.exit("[ERROR] Test split is empty; adjust ratios")

# Fit scaler on train only (strict leakage control)
sc = StandardScaler()
train_df[FEATS] = sc.fit_transform(train_df[FEATS]).astype(np.float32)
val_df[FEATS] = sc.transform(val_df[FEATS]).astype(np.float32)
test_df[FEATS] = sc.transform(test_df[FEATS]).astype(np.float32)

X, y = build_seq(train_df, FEATS, args.window)
if X.shape[0] == 0:
    sys.exit(f"[ERROR] Train rows are insufficient for window={args.window}")

ds = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
dl = torch.utils.data.DataLoader(ds, args.batch, shuffle=True)

model = LSTMDir(len(FEATS), att=args.use_attn).to(DEVICE)
pos_ratio = y.mean(); neg_ratio = 1 - pos_ratio
safe_pos = max(float(pos_ratio), 1e-6)
crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg_ratio/safe_pos]).to(DEVICE))
opt  = torch.optim.Adam(model.parameters(), args.lr)

best, wait = math.inf, 0
for ep in range(args.epochs):
    model.train(); loss_sum = 0.0
    for xb, yb in dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(xb), yb); loss.backward(); opt.step()
        loss_sum += loss.item() * len(xb)
    avg = loss_sum / len(ds)
    print(f"[{ep+1:03d}] loss={avg:.4f}")
    if avg < best:
        best, wait = avg, 0
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
val_metrics = evaluate_split(best_model, val_df, FEATS, args.window, DEVICE, args.eval_threshold)
test_metrics = evaluate_split(best_model, test_df, FEATS, args.window, DEVICE, args.eval_threshold)

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

meta = {
    "seed": args.seed,
    "window": args.window,
    "eval_threshold": args.eval_threshold,
    "features": FEATS,
    "use_attn": args.use_attn,
    "train_ratio": args.train_ratio,
    "val_ratio": args.val_ratio,
    "test_ratio": 1 - args.train_ratio - args.val_ratio,
    "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    "metrics": {
        "val": val_metrics,
        "test": test_metrics,
    },
    "model_path": str(save_path),
    "scaler_path": str(scaler_path),
}
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"[OK] saved scaler to {scaler_path}")
print(f"[OK] saved metadata to {meta_path}")
