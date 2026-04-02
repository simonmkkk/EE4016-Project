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
from sklearn.metrics import accuracy_score, f1_score

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
ap.add_argument("--protocol", type=str, help="Path to experiment protocol json for baseline parameters")
ap.add_argument("--eval_split", choices=["all", "test"], default="test", help="Evaluate on full data or unseen test split")
args = ap.parse_args()


def infer_periods_per_year(csv_name: str) -> float:
    name = csv_name.lower()
    if "_1d" in name or "_5d" in name:
        return 252.0
    if "_1h" in name or "_60m" in name:
        return 252.0 * 6.5
    if "_30m" in name:
        return 252.0 * 13.0
    if "_15m" in name:
        return 252.0 * 26.0
    if "_5m" in name:
        return 252.0 * 78.0
    if "_1m" in name:
        return 252.0 * 390.0
    return 252.0

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

protocol = {}
if args.protocol:
    with open(args.protocol, "r", encoding="utf-8") as f:
        protocol = json.load(f)

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

fwd_ret = df["fwd_ret"].iloc[args.window:].values.astype(np.float64)
bp = protocol.get("baseline_params", {})
macd_p = bp.get("macd", {"fast": 12, "slow": 26, "signal": 9})
rsi_p = bp.get("rsi", {"window": 14, "oversold": 30, "overbought": 70})
bb_p = bp.get("bollinger", {"window": 20, "std": 2.0})

macd_obj = MACD(
    df["close"],
    window_fast=int(macd_p.get("fast", 12)),
    window_slow=int(macd_p.get("slow", 26)),
    window_sign=int(macd_p.get("signal", 9)),
)
rsi_obj = RSIIndicator(df["close"], window=int(rsi_p.get("window", 14)))
bb_obj = BollingerBands(
    df["close"],
    window=int(bb_p.get("window", 20)),
    window_dev=float(bb_p.get("std", 2.0)),
)
macd_line = macd_obj.macd().iloc[args.window:].fillna(0).values
macd_signal = macd_obj.macd_signal().iloc[args.window:].fillna(0).values
rsi_vals = rsi_obj.rsi().iloc[args.window:].fillna(50).values
bb_high = bb_obj.bollinger_hband().iloc[args.window:].bfill().fillna(df["close"].iloc[args.window:]).values
bb_low = bb_obj.bollinger_lband().iloc[args.window:].bfill().fillna(df["close"].iloc[args.window:]).values
close_eval = df["close"].iloc[args.window:].values

if args.eval_split == "test":
    tr = float(meta.get("train_ratio", 0.7))
    vr = float(meta.get("val_ratio", 0.15))
    start = int(len(fwd_ret) * (tr + vr))
else:
    start = 0

fwd_ret = fwd_ret[start:]
probs = probs[start:]
macd_line = macd_line[start:]
macd_signal = macd_signal[start:]
rsi_vals = rsi_vals[start:]
bb_high = bb_high[start:]
bb_low = bb_low[start:]
close_eval = close_eval[start:]
periods_per_year = infer_periods_per_year(Path(args.csv).name)

def strategy_metrics(signals: np.ndarray):
    turnover = np.abs(np.diff(np.insert(signals, 0, 0))).astype(np.float64)
    strat_ret = signals * fwd_ret - args.fee * turnover
    equity = np.cumprod(1.0 + strat_ret) if len(strat_ret) else np.array([], dtype=np.float64)
    roll_max = np.maximum.accumulate(equity) if len(equity) else np.array([], dtype=np.float64)
    max_dd = float(np.min(equity / roll_max - 1.0)) if len(equity) else 0.0
    sharpe = float(np.sqrt(periods_per_year) * strat_ret.mean() / (strat_ret.std() + 1e-12)) if len(strat_ret) else 0.0
    non_zero = strat_ret[strat_ret != 0]
    win_rate = float((non_zero > 0).mean()) if len(non_zero) else 0.0
    turnover_rate = float((turnover > 0).mean()) if len(turnover) else 0.0
    return strat_ret, equity, turnover, {
        "n_trades": int(np.sum(turnover)),
        "total_return": float(equity[-1] - 1.0) if len(equity) else 0.0,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "avg_turnover": float(turnover.mean()) if len(turnover) else 0.0,
        "turnover_rate": turnover_rate,
    }

signals_model = np.where(probs > args.threshold, 1, -1).astype(np.int8)
signals_bh = np.ones_like(signals_model, dtype=np.int8)
signals_macd = np.where(macd_line > macd_signal, 1, -1).astype(np.int8)
signals_rsi = np.where(rsi_vals < float(rsi_p.get("oversold", 30)), 1, np.where(rsi_vals > float(rsi_p.get("overbought", 70)), -1, 0)).astype(np.int8)
signals_bb = np.where(close_eval < bb_low, 1, np.where(close_eval > bb_high, -1, 0)).astype(np.int8)

model_ret, equity, turnover, model_summary = strategy_metrics(signals_model)
bh_ret, _, _, bh_summary = strategy_metrics(signals_bh)
macd_ret, _, _, macd_summary = strategy_metrics(signals_macd)
rsi_ret, _, _, rsi_summary = strategy_metrics(signals_rsi)
bb_ret, _, _, bb_summary = strategy_metrics(signals_bb)

true_up = (fwd_ret > 0).astype(int)
pred_up = (signals_model > 0).astype(int)
predictive_metrics = {
    "accuracy": float(accuracy_score(true_up, pred_up)) if len(true_up) else 0.0,
    "f1": float(f1_score(true_up, pred_up, zero_division=0)) if len(true_up) else 0.0,
}

out = pd.DataFrame(
    {
        "pred_prob": np.round(probs, 6),
        "signal": signals_model,
        "fwd_ret": fwd_ret,
        "turnover": turnover,
        "strategy_ret": model_ret,
        "equity": equity,
        "bh_ret": bh_ret,
        "macd_ret": macd_ret,
        "rsi_ret": rsi_ret,
        "bb_ret": bb_ret,
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
    "eval_split": args.eval_split,
    "protocol_baseline_params": {
        "macd": macd_p,
        "rsi": rsi_p,
        "bollinger": bb_p,
    },
    "strategies": {
        "model_lstm": model_summary,
        "buy_and_hold": bh_summary,
        "macd": macd_summary,
        "rsi": rsi_summary,
        "bollinger": bb_summary,
    },
    "predictive_metrics_model": predictive_metrics,
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
    f"[BT] model_return={summary['strategies']['model_lstm']['total_return']:.4f} "
    f"model_acc={summary['predictive_metrics_model']['accuracy']:.4f} "
    f"model_f1={summary['predictive_metrics_model']['f1']:.4f} "
    f"bh_return={summary['strategies']['buy_and_hold']['total_return']:.4f} "
    f"macd_return={summary['strategies']['macd']['total_return']:.4f} "
    f"rsi_return={summary['strategies']['rsi']['total_return']:.4f} "
    f"bb_return={summary['strategies']['bollinger']['total_return']:.4f}"
)
