#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
backtest_stock.py -- leakage-safe backtest using saved model/scaler/meta artifacts.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fe(df: pd.DataFrame):
    df = df.copy()
    df["rsi"] = RSIIndicator(df["close"]).rsi()
    df["macd"] = MACD(df["close"]).macd_diff()
    df["bbw"] = BollingerBands(df["close"]).bollinger_wband()
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
    df["vma20"] = SMAIndicator(df["volume"], 20).sma_indicator()
    df["v_ratio"] = df["volume"] / df["vma20"]
    df["body"] = (df["open"] - df["close"]).abs()
    df["range"] = df["high"] - df["low"]
    df["fwd_ret"] = df["close"].pct_change().shift(-1)
    return df.dropna().reset_index(drop=True)


def build_seq(frame, feats, window):
    X = []
    v = frame[feats].values.astype(np.float32)
    for i in range(window, len(frame)):
        X.append(v[i - window : i])
    return np.array(X)


class LSTMDir(nn.Module):
    def __init__(self, d_in: int, hid: int = 128, att: bool = False):
        super().__init__()
        self.att = att
        self.lstm = nn.LSTM(d_in, hid, batch_first=True)
        if att:
            self.w = nn.Linear(hid, 1, bias=False)
        self.fc = nn.Linear(hid, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        o = (torch.softmax(self.w(o), 1) * o).sum(1) if self.att else o[:, -1]
        return self.fc(o)


ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--meta", help="default: same basename as model")
ap.add_argument("--scaler", help="default: same basename as model")
ap.add_argument("--window", type=int, default=30)
ap.add_argument("--threshold", type=float, default=None)
ap.add_argument("--fee", type=float, default=0.001, help="transaction fee per position change")
ap.add_argument("--out", help="output csv filename")
args = ap.parse_args()

model_path = Path(args.model)
meta_path = Path(args.meta) if args.meta else model_path.with_suffix(".meta.json")
scaler_path = Path(args.scaler) if args.scaler else model_path.with_suffix(".scaler.pkl")
if not meta_path.exists():
    raise FileNotFoundError(f"Metadata not found: {meta_path}")
if not scaler_path.exists():
    raise FileNotFoundError(f"Scaler not found: {scaler_path}")

with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

if args.threshold is None:
    args.threshold = float(meta.get("eval_threshold", 0.5))
if args.window == 30 and isinstance(meta.get("window"), int):
    args.window = int(meta["window"])

df0 = pd.read_csv(args.csv, parse_dates=["date"])
df0["date"] = pd.to_datetime(df0["date"], utc=True).dt.tz_localize(None)
df0["granularity"] = (
    (df0["date"].dt.hour != 0) | (df0["date"].dt.minute != 0) | (df0["date"].dt.second != 0)
).astype(np.int8)
df = fe(df0.sort_values("date"))

feats = meta.get("features", [])
missing = [c for c in feats if c not in df.columns]
if missing:
    raise ValueError(f"Missing features required by metadata: {missing}")

df[feats] = scaler.transform(df[feats]).astype(np.float32)
X = build_seq(df, feats, args.window)
if X.shape[0] == 0:
    raise ValueError(f"Insufficient rows for window={args.window}")

model = LSTMDir(len(feats), att=bool(meta.get("use_attn", False))).to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(torch.tensor(X).to(DEVICE))).cpu().numpy().flatten()

signals = np.where(probs > args.threshold, 1, -1).astype(np.int8)
fwd_ret = df["fwd_ret"].iloc[args.window:].values.astype(np.float64)
turnover = np.abs(np.diff(np.insert(signals, 0, 0))).astype(np.float64)
strategy_ret = signals * fwd_ret - args.fee * turnover
equity = np.cumprod(1.0 + strategy_ret)
roll_max = np.maximum.accumulate(equity)
max_drawdown = float(np.min(equity / roll_max - 1.0)) if len(equity) else 0.0
sharpe = float(np.sqrt(252) * strategy_ret.mean() / (strategy_ret.std() + 1e-12)) if len(strategy_ret) else 0.0

out = pd.DataFrame(
    {
        "pred_prob": np.round(probs, 6),
        "signal": signals,
        "fwd_ret": fwd_ret,
        "turnover": turnover,
        "strategy_ret": strategy_ret,
        "equity": equity,
    }
)

symbol = Path(args.csv).stem.split("_")[0].upper()
result_dir = Path("result") / symbol
result_dir.mkdir(parents=True, exist_ok=True)
out_name = Path(args.out).name if args.out else f"{Path(args.csv).stem}_bt.csv"
out_path = result_dir / out_name
out.to_csv(out_path, index=False, encoding="utf-8-sig")

summary = {
    "threshold": args.threshold,
    "fee": args.fee,
    "n_trades": int(np.sum(turnover)),
    "total_return": float(equity[-1] - 1.0) if len(equity) else 0.0,
    "sharpe": sharpe,
    "max_drawdown": max_drawdown,
    "model_path": str(model_path),
    "meta_path": str(meta_path),
    "scaler_path": str(scaler_path),
}
summary_path = result_dir / f"{Path(args.csv).stem}_bt_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"[OK] backtest saved to {out_path}")
print(f"[OK] summary saved to {summary_path}")
print(
    f"[BT] total_return={summary['total_return']:.4f} "
    f"sharpe={summary['sharpe']:.4f} mdd={summary['max_drawdown']:.4f} trades={summary['n_trades']}"
)
