#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_stock.py  ── 推論 + 解釋版
  ‧ 支援日/分鐘線混訓模型
  ‧ 標記高信心卻錯誤
  ‧ 為每筆結果產生 explanation / why_wrong / improve_tip
"""
import sys, argparse, json, pickle, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    X = []
    v = frame[feats].values.astype(np.float32)
    for i in range(window, len(frame)):
        X.append(v[i-window:i])
    return np.array(X)

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
        o = (torch.softmax(self.w(o),1)*o).sum(1) if self.att else o[:, -1]
        return self.fc(o)

# ╭────────────── CLI & 互動模式 ─────────────╮
ap = argparse.ArgumentParser()
if len(sys.argv) == 1:  # ---- Interactive ----
    print("\n=== Interactive mode ===")
    ipt = lambda msg, d='': input(f"{msg} [{d}] ").strip() or d
    sys.argv += [
        "--csv",          ipt("CSV 路徑"),
        "--model",        ipt("模型 (.pt) 路徑"),
        "--window",       ipt("window", '30'),
        "--threshold",    ipt("threshold", '0.4'),
        "--conf_thresh",  ipt("high-conf 閾值", '0.8'),
    ]
    if input("使用 Attention? (y/n) [n] ").lower().startswith('y'):
        sys.argv.append("--use_attn")
    out_ = input("輸出檔名 (留空自動命名): ").strip()
    if out_:
        sys.argv += ["--out", out_]

ap.add_argument("--csv",   required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--window", type=int, default=30)
ap.add_argument("--threshold", type=float, default=0.4)
ap.add_argument("--conf_thresh", type=float, default=0.8,
                help="若 pred_prob ≥ conf_thresh 且預測錯，標記 high_conf_wrong")
ap.add_argument("--use_attn", action="store_true")
ap.add_argument("--out", help="輸出檔名 (default 自動)")
ap.add_argument("--scaler", help="scaler 檔路徑 (.pkl), default: 與 model 同名 .scaler.pkl")
ap.add_argument("--meta", help="metadata 路徑 (.json), default: 與 model 同名 .meta.json")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)

# Always save under result/{stock-symbol}/
symbol = Path(args.csv).stem.split("_")[0].upper()
result_dir = Path("result") / symbol
result_dir.mkdir(parents=True, exist_ok=True)

# ╭────────── 讀檔 + granularity ───────────╮
df0 = pd.read_csv(args.csv, parse_dates=["date"])
df0["date"] = pd.to_datetime(df0["date"], utc=True).dt.tz_localize(None)
df0["granularity"] = ((df0["date"].dt.hour != 0) |
                      (df0["date"].dt.minute != 0) |
                      (df0["date"].dt.second != 0)).astype(np.int8)
df = fe(df0.sort_values("date"))

FEATS = [c for c in df.columns if c not in ["date", "log_ret", "direction"]]
if "granularity" not in FEATS:
    FEATS.append("granularity")

# Load metadata/scaler produced by training (strict leakage control)
model_path = Path(args.model)
meta_path = Path(args.meta) if args.meta else model_path.with_suffix(".meta.json")
scaler_path = Path(args.scaler) if args.scaler else model_path.with_suffix(".scaler.pkl")

trained_feats = FEATS
if meta_path.exists():
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    trained_feats = meta.get("features", FEATS)
    if not args.use_attn:
        args.use_attn = bool(meta.get("use_attn", False))
    if args.window == 30 and isinstance(meta.get("window"), int):
        args.window = meta["window"]

missing_feats = [c for c in trained_feats if c not in df.columns]
if missing_feats:
    sys.exit(f"[ERROR] Missing required features from metadata: {missing_feats}")
FEATS = trained_feats

if not scaler_path.exists():
    sys.exit(f"[ERROR] scaler file not found: {scaler_path}")
with open(scaler_path, "rb") as f:
    sc = pickle.load(f)
df[FEATS] = sc.transform(df[FEATS]).astype(np.float32)
X = build_seq(df, FEATS, args.window)
if X.shape[0] == 0:
    sys.exit(f"[ERROR] Data rows are insufficient for window={args.window}")

# ╭────────── Inference ─────────────────────╮
model = LSTMDir(len(FEATS), att=args.use_attn).to(DEVICE)
model.load_state_dict(torch.load(args.model, map_location=DEVICE)); model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(torch.tensor(X).to(DEVICE))).cpu().numpy().flatten()

preds   = (probs > args.threshold).astype(int)
actual  = df["direction"].iloc[args.window:].astype(int).values
correct = preds == actual
high_conf_wrong = (probs >= args.conf_thresh) & (~correct)

# ╭────────── Explanation helpers ───────────╮
# ─── Explanation helpers ───
def gen_explanation(feat_row, pred):
    rs, mc, vr = feat_row["rsi"], feat_row["macd"], feat_row["v_ratio"]
    reasons = []
    if rs > 70:  reasons.append("RSI>70 (超買)")
    elif rs < 30: reasons.append("RSI<30 (超賣)")
    if mc > 0:   reasons.append("MACD 正")
    elif mc < 0: reasons.append("MACD 負")
    if vr > 1.5: reasons.append("成交量放大")
    direction = "上漲" if pred else "下跌"
    return f"判斷{direction}: " + ("；".join(reasons) if reasons else "無明顯指標")

def why_wrong(feat_row, is_correct):          # ← 只看布林 is_correct
    if is_correct:
        return ""
    tips = []
    bull = (feat_row["rsi"] > 55) + (feat_row["macd"] > 0)
    bear = (feat_row["rsi"] < 45) + (feat_row["macd"] < 0)
    if bull and bear:
        tips.append("指標彼此矛盾")
    if feat_row["atr"] > 2.5:
        tips.append("ATR 異常高")
    if not tips:
        tips.append("模型閾值或特徵不足")
    return "；".join(tips)

def improve_tip(feat_row, is_correct):        # ← 同理改用 is_correct
    if is_correct:
        return ""
    adv = []
    if abs(feat_row["rsi"] - 50) < 5:
        adv.append("調整 RSI 閾值")
    if abs(feat_row["macd"]) < 0.05:
        adv.append("加入趨勢／動能特徵")
    if feat_row["atr"] > 2.5:
        adv.append("用波動度動態 threshold")
    if not adv:
        adv.append("微調模型參數或擴增資料")
    return "；".join(adv)


# 將對應特徵對齊 out_df 的索引
feat_part = df.iloc[args.window:].reset_index(drop=True)

out_df = pd.DataFrame({
    "pred_prob": np.round(probs, 2),
    "prediction": preds,
    "actual": actual,
    "correct": correct,
    "high_conf_wrong": high_conf_wrong
})

exps, whys, tips = [], [], []
for i in range(len(out_df)):
    feat_row = feat_part.iloc[i]
    exps.append(gen_explanation(feat_row, preds[i]))
    whys.append(why_wrong(feat_row, correct[i]))
    tips.append(improve_tip(feat_row, correct[i]))

out_df["explanation"] = exps
out_df["why_wrong"]   = whys
out_df["improve_tip"] = tips


# ╭────────── Print & Save ──────────────────╮
print("\nLast 5 predictions (with explanation):")
print(out_df.tail(5).to_string(index=False, max_colwidth=60))
acc = correct.mean()*100
print(f"\nAccuracy = {acc:.2f}%")

out_filename = (Path(args.out).name if args.out else f"{Path(args.csv).stem}_pred.csv")
out_path = result_dir / out_filename
out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
with open(out_path, "a", encoding="utf-8-sig") as f:
    f.write(f"\naccuracy,,,{acc:.2f}%\n")
print(f"[OK] saved to {out_path}")

bad_rows = out_df[out_df["high_conf_wrong"]]
if not bad_rows.empty:
    bad_path = result_dir / f"{Path(args.csv).stem}_bad.csv"
    bad_rows.to_csv(bad_path, index=False, encoding="utf-8-sig")
    print(f"[WARN] High-conf wrong: {len(bad_rows)} rows, saved to {bad_path}")
