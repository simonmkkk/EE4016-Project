#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_stock.py -- inference with explanations.

Supports models trained on mixed daily / intraday series, flags high-confidence
errors, and adds explanation / why_wrong / improve_tip columns per row.
"""
import sys, argparse, json, pickle, random
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from app.paths import RESULTS_DIR
from app.fe import add_technical_indicators
from app.constants import interval_id_from_csv_stem

import numpy as np
import pandas as pd
import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Feature engineering ---
def fe(df: pd.DataFrame):
    out = add_technical_indicators(df)
    out["log_ret"] = np.log(out["close"]).diff().shift(-1)
    out["direction"] = (out["log_ret"] > 0).astype(np.float32)
    return out.dropna().reset_index(drop=True)

def build_seq(frame, feats, window):
    X = []
    v = frame[feats].values.astype(np.float32)
    for i in range(window, len(frame)):
        X.append(v[i-window:i])
    return np.array(X)

# --- Model ---
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

# --- CLI & interactive mode ---
ap = argparse.ArgumentParser()
if len(sys.argv) == 1:  # ---- Interactive ----
    print("\n=== Interactive mode ===")
    ipt = lambda msg, d='': input(f"{msg} [{d}] ").strip() or d
    csv_path = ipt("CSV path")
    model_path = ipt("Model (.pt) path")
    window = ipt("window", "30")
    threshold = input("threshold (blank = read eval_threshold from metadata) [] ").strip()
    conf_thresh = ipt("high-confidence threshold", "0.8")
    sys.argv += ["--csv", csv_path, "--model", model_path, "--window", window, "--conf_thresh", conf_thresh]
    if threshold:
        sys.argv += ["--threshold", threshold]
    if input("Use attention? (y/n) [n] ").lower().startswith("y"):
        sys.argv.append("--use_attn")
    out_ = input("Output filename (blank for auto name): ").strip()
    if out_:
        sys.argv += ["--out", out_]

ap.add_argument("--csv",   required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--window", type=int, default=30)
ap.add_argument("--threshold", type=float, default=None)
ap.add_argument(
    "--conf_thresh",
    type=float,
    default=0.8,
    help="if pred_prob >= conf_thresh and prediction is wrong, set high_conf_wrong",
)
ap.add_argument("--use_attn", action="store_true")
ap.add_argument("--out", help="output CSV basename (default: auto from input CSV stem)")
ap.add_argument(
    "--scaler",
    help="path to scaler .pkl (default: same basename as model with .scaler.pkl)",
)
ap.add_argument(
    "--meta",
    help="path to metadata .json (default: same basename as model with .meta.json)",
)
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

# Always save under RESULTS_DIR/{stock-symbol}/ (.env RESULTS_DIR)
symbol = Path(args.csv).stem.split("_")[0].upper()
result_dir = RESULTS_DIR / symbol
result_dir.mkdir(parents=True, exist_ok=True)

# --- Load CSV + granularity flag ---
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
meta = None
if meta_path.exists():
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    trained_feats = meta.get("features", FEATS)
    if not args.use_attn:
        args.use_attn = bool(meta.get("use_attn", False))
    if args.window == 30 and isinstance(meta.get("window"), int):
        args.window = meta["window"]

if args.threshold is None:
    if isinstance(meta, dict) and isinstance(meta.get("eval_threshold"), (int, float)):
        args.threshold = float(meta["eval_threshold"])
    else:
        args.threshold = 0.4

if "interval_id" in trained_feats:
    df["interval_id"] = np.float32(interval_id_from_csv_stem(Path(args.csv).stem))

missing_feats = [c for c in trained_feats if c not in df.columns]
if missing_feats:
    sys.exit(f"[ERROR] Missing required features from metadata: {missing_feats}")
FEATS = trained_feats

if not scaler_path.exists():
    sys.exit(f"[ERROR] scaler file not found: {scaler_path}")
with open(scaler_path, "rb") as f:
    sc = pickle.load(f)
scaled_feats = (
    meta.get("features_scaled")
    if isinstance(meta, dict) and isinstance(meta.get("features_scaled"), list)
    else (
        [c for c in trained_feats if c != "interval_id"]
        if "interval_id" in trained_feats
        else list(trained_feats)
    )
)
df[scaled_feats] = sc.transform(df[scaled_feats]).astype(np.float32)
X = build_seq(df, FEATS, args.window)
if X.shape[0] == 0:
    sys.exit(f"[ERROR] Data rows are insufficient for window={args.window}")

# --- Inference ---
model = LSTMDir(len(FEATS), att=args.use_attn).to(DEVICE)
model.load_state_dict(torch.load(args.model, map_location=DEVICE)); model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(torch.tensor(X).to(DEVICE))).cpu().numpy().flatten()

preds   = (probs > args.threshold).astype(int)
actual  = df["direction"].iloc[args.window:].astype(int).values
correct = preds == actual
high_conf_wrong = (probs >= args.conf_thresh) & (~correct)

# --- Explanation helpers ---
def gen_explanation(feat_row, pred):
    rs, mc, vr = feat_row["rsi"], feat_row["macd"], feat_row["v_ratio"]
    reasons = []
    if rs > 70:
        reasons.append("RSI>70 (overbought)")
    elif rs < 30:
        reasons.append("RSI<30 (oversold)")
    if mc > 0:
        reasons.append("MACD positive")
    elif mc < 0:
        reasons.append("MACD negative")
    if vr > 1.5:
        reasons.append("volume spike (v_ratio)")
    direction = "up" if pred else "down"
    return f"{direction} bias: " + ("; ".join(reasons) if reasons else "no strong signal")

def why_wrong(feat_row, is_correct):
    if is_correct:
        return ""
    tips = []
    bull = (feat_row["rsi"] > 55) + (feat_row["macd"] > 0)
    bear = (feat_row["rsi"] < 45) + (feat_row["macd"] < 0)
    if bull and bear:
        tips.append("mixed bullish/bearish indicators")
    if feat_row["atr"] > 2.5:
        tips.append("ATR unusually high")
    if not tips:
        tips.append("threshold or features insufficient")
    return "; ".join(tips)

def improve_tip(feat_row, is_correct):
    if is_correct:
        return ""
    adv = []
    if abs(feat_row["rsi"] - 50) < 5:
        adv.append("tune RSI bands")
    if abs(feat_row["macd"]) < 0.05:
        adv.append("add trend/momentum features")
    if feat_row["atr"] > 2.5:
        adv.append("use volatility-aware threshold")
    if not adv:
        adv.append("tune model or augment data")
    return "; ".join(adv)


# Align feature rows with out_df indices (after window offset)
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


# --- Print and save ---
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
