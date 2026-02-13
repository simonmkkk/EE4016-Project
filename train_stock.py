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
import os, glob, sys, math, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

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
ap.add_argument("--save_model", required=True)
ap.add_argument("--window", type=int, default=30)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch",  type=int, default=256)
ap.add_argument("--lr",     type=float, default=1e-3)
ap.add_argument("--patience", type=int, default=6)
ap.add_argument("--use_attn", action="store_true")

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
    csv_list += glob.glob(os.path.join(args.csv_dir, "*.csv"))
if not csv_list:
    sys.exit("[ERROR] Must specify --csvs or --csv_dir")

frames = [read_and_fe(p) for p in csv_list]           # ★ NEW
data   = pd.concat(frames).reset_index(drop=True)

FEATS = [c for c in data.columns if c not in ["date", "log_ret", "direction"]]
if "granularity" not in FEATS:                        # ★ NEW
    FEATS.append("granularity")

sc = StandardScaler()
data[FEATS] = sc.fit_transform(data[FEATS]).astype(np.float32)

X, y = build_seq(data, FEATS, args.window)
if X.shape[0] == 0:
    sys.exit(f"資料行數不足 (≤{args.window})")

ds = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
dl = torch.utils.data.DataLoader(ds, args.batch, shuffle=True)

model = LSTMDir(len(FEATS), att=args.use_attn).to(DEVICE)
pos_ratio = y.mean(); neg_ratio = 1 - pos_ratio
crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg_ratio/pos_ratio]).to(DEVICE))
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
        torch.save(model.state_dict(), args.save_model)
    else:
        wait += 1
        if wait >= args.patience:
            print("[early stop]"); break
print(f"[OK] saved to {args.save_model}")
