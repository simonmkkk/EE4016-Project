from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..domain.datasets import infer_symbol_from_path, read_prediction_frame
from ..infra.artifacts import (
    build_model_from_meta,
    load_artifact_bundle,
    scaled_feature_columns,
)
from ..model import lstm_feature_columns
from ..paths import RESULTS_DIR
from ..sequences import build_seq_x_only


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.8,
        help="if pred_prob >= conf_thresh and prediction is wrong, set high_conf_wrong",
    )
    parser.add_argument("--use_attn", action="store_true")
    parser.add_argument(
        "--out",
        help="output CSV basename (default: auto from input CSV stem)",
    )
    parser.add_argument(
        "--scaler",
        help="path to scaler .pkl (default: same basename as model with .scaler.pkl)",
    )
    parser.add_argument(
        "--meta",
        help="path to metadata .json (default: same basename as model with .meta.json)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--infer_batch",
        type=int,
        default=256,
        help="Inference batch size (avoids GPU OOM on long CSVs).",
    )
    return parser


def _interactive_args() -> list[str]:
    print("\n=== Predict - Interactive mode ===")
    csv_path = input("CSV path [] ").strip()
    model_path = input("Model (.pt) path [] ").strip()
    window = input("window [30] ").strip() or "30"
    threshold = input("threshold (blank = read eval_threshold from metadata) [] ").strip()
    conf_thresh = input("high-confidence threshold [0.8] ").strip() or "0.8"
    args = [
        "--csv",
        csv_path,
        "--model",
        model_path,
        "--window",
        window,
        "--conf_thresh",
        conf_thresh,
    ]
    if threshold:
        args += ["--threshold", threshold]
    if input("Use attention? (y/n) [n] ").lower().startswith("y"):
        args.append("--use_attn")
    out_name = input("Output filename (blank for auto name): ").strip()
    if out_name:
        args += ["--out", out_name]
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    cli_args = list(argv) if argv is not None else None
    if cli_args is None:
        import sys

        cli_args = sys.argv[1:]
    if not cli_args:
        cli_args = _interactive_args()
    return build_parser().parse_args(cli_args)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def gen_explanation(feat_row: pd.Series, pred: int) -> str:
    reasons = []
    if feat_row["rsi"] > 70:
        reasons.append("RSI>70 (overbought)")
    elif feat_row["rsi"] < 30:
        reasons.append("RSI<30 (oversold)")
    if feat_row["macd"] > 0:
        reasons.append("MACD positive")
    elif feat_row["macd"] < 0:
        reasons.append("MACD negative")
    if feat_row["v_ratio"] > 1.5:
        reasons.append("volume spike (v_ratio)")
    direction = "up" if pred else "down"
    return f"{direction} bias: " + ("; ".join(reasons) if reasons else "no strong signal")


def why_wrong(feat_row: pd.Series, is_correct: bool, labeled: bool) -> str:
    if not labeled or is_correct:
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


def improve_tip(feat_row: pd.Series, is_correct: bool, labeled: bool) -> str:
    if not labeled or is_correct:
        return ""
    advice = []
    if abs(feat_row["rsi"] - 50) < 5:
        advice.append("tune RSI bands")
    if abs(feat_row["macd"]) < 0.05:
        advice.append("add trend/momentum features")
    if feat_row["atr"] > 2.5:
        advice.append("use volatility-aware threshold")
    if not advice:
        advice.append("tune model or augment data")
    return "; ".join(advice)


def run_predict(args: argparse.Namespace) -> int:
    set_seed(args.seed)

    symbol = infer_symbol_from_path(args.csv)
    result_dir = RESULTS_DIR / symbol
    result_dir.mkdir(parents=True, exist_ok=True)

    df = read_prediction_frame(args.csv, keep_last_bar=True)
    features = [c for c in df.columns if c not in ["date", "log_ret", "direction"]]
    if "granularity" not in features:
        features.append("granularity")

    bundle = load_artifact_bundle(
        args.model,
        meta_path=args.meta,
        scaler_path=args.scaler,
    )
    meta = bundle.meta
    trained_features = meta.get("features", features)
    if not args.use_attn:
        args.use_attn = bool(meta.get("use_attn", False))
    if args.window == 30 and isinstance(meta.get("window"), int):
        args.window = int(meta["window"])
    if args.threshold is None:
        if isinstance(meta.get("eval_threshold"), (int, float)):
            args.threshold = float(meta["eval_threshold"])
        else:
            args.threshold = 0.4

    use_interval_embedding = bool(meta.get("use_interval_embedding", False))
    if isinstance(meta.get("feats_lstm"), list):
        feats_lstm = meta["feats_lstm"]
    else:
        feats_lstm, _ = lstm_feature_columns(trained_features, use_interval_embedding)

    missing_features = [col for col in trained_features if col not in df.columns]
    if missing_features:
        raise SystemExit(f"[ERROR] Missing required features from metadata: {missing_features}")

    scaled_features = scaled_feature_columns(meta, trained_features)
    df[scaled_features] = bundle.scaler.transform(df[scaled_features]).astype(np.float32)

    X, iv, x_next, iv_next = build_seq_x_only(
        df,
        feats_lstm,
        args.window,
        use_interval_embedding=use_interval_embedding,
        extra_next_bar=True,
    )
    if X.shape[0] == 0:
        raise SystemExit(f"[ERROR] Data rows are insufficient for window={args.window}")

    model = build_model_from_meta(
        len(feats_lstm),
        meta,
        device=DEVICE,
        use_attn=args.use_attn,
    )
    model.load_state_dict(torch.load(bundle.model_path, map_location=DEVICE, weights_only=True))
    model.eval()

    infer_bs = max(1, int(args.infer_batch))
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, X.shape[0], infer_bs):
            end = min(start + infer_bs, X.shape[0])
            xb = torch.tensor(X[start:end]).to(DEVICE)
            if iv is not None:
                ivb = torch.tensor(iv[start:end]).to(DEVICE)
                probs_batch = torch.sigmoid(model(xb, ivb)).cpu().numpy().reshape(-1)
            else:
                probs_batch = torch.sigmoid(model(xb)).cpu().numpy().reshape(-1)
            chunks.append(probs_batch)
    probs = np.concatenate(chunks, axis=0)

    preds = (probs > args.threshold).astype(int)
    actual = df["direction"].iloc[args.window:].values.astype(float)
    has_label = ~np.isnan(actual)
    correct = np.zeros(len(preds), dtype=bool)
    correct[has_label] = preds[has_label] == actual[has_label].astype(int)
    high_conf_wrong = np.zeros(len(preds), dtype=bool)
    high_conf_wrong[has_label] = (probs[has_label] >= args.conf_thresh) & (
        preds[has_label] != actual[has_label].astype(int)
    )

    prob_next = None
    if x_next is not None:
        with torch.no_grad():
            xn = torch.tensor(x_next).to(DEVICE)
            if iv_next is not None:
                prob_next = float(
                    torch.sigmoid(model(xn, torch.tensor(iv_next).to(DEVICE))).cpu().item()
                )
            else:
                prob_next = float(torch.sigmoid(model(xn)).cpu().item())

    feat_part = df.iloc[args.window:].reset_index(drop=True)
    out_df = pd.DataFrame(
        {
            "pred_prob": np.round(probs, 2),
            "prediction": preds,
            "actual": actual,
            "correct": correct,
            "high_conf_wrong": high_conf_wrong,
        }
    )

    explanations: list[str] = []
    why_rows: list[str] = []
    tips: list[str] = []
    for idx in range(len(out_df)):
        feat_row = feat_part.iloc[idx]
        labeled = bool(has_label[idx])
        explanations.append(gen_explanation(feat_row, preds[idx]))
        why_rows.append(why_wrong(feat_row, bool(correct[idx]), labeled))
        tips.append(improve_tip(feat_row, bool(correct[idx]), labeled))

    out_df["explanation"] = explanations
    out_df["why_wrong"] = why_rows
    out_df["improve_tip"] = tips

    print("\nLast 5 predictions (with explanation):")
    print(out_df.tail(5).to_string(index=False, max_colwidth=60))
    if prob_next is not None:
        print(
            "\nNext-bar forecast (after last close, label unknown): "
            f"P(up)={prob_next:.4f}  pred={'up' if prob_next > args.threshold else 'down'}"
        )
    acc = float(correct[has_label].mean() * 100) if has_label.any() else 0.0
    print(f"\nAccuracy (labeled rows only) = {acc:.2f}%")

    out_filename = Path(args.out).name if args.out else f"{Path(args.csv).stem}_pred.csv"
    out_path = result_dir / out_filename
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    with open(out_path, "a", encoding="utf-8-sig") as f:
        f.write(f"\naccuracy_labeled_rows_only,,,{acc:.2f}%\n")
        if prob_next is not None:
            f.write(f"next_bar_forecast_prob,,,{prob_next:.6f}\n")
    print(f"[OK] saved to {out_path}")

    bad_rows = out_df[out_df["high_conf_wrong"]]
    if not bad_rows.empty:
        bad_path = result_dir / f"{Path(args.csv).stem}_bad.csv"
        bad_rows.to_csv(bad_path, index=False, encoding="utf-8-sig")
        print(f"[WARN] High-conf wrong: {len(bad_rows)} rows, saved to {bad_path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_predict(args)


if __name__ == "__main__":
    raise SystemExit(main())
