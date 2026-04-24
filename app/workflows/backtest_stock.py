from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

from ..domain.datasets import infer_symbol_from_path, read_backtest_frame
from ..infra.artifacts import (
    build_model_from_meta,
    load_artifact_bundle,
    scaled_feature_columns,
)
from ..infra.files import dump_json, load_json
from ..model import lstm_feature_columns
from ..paths import RESULTS_DIR, SAVE_DIR
from ..sequences import build_seq_x_only


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LINE_WIDTH = 70


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv")
    parser.add_argument("--model")
    parser.add_argument("--meta", help="default: same basename as model")
    parser.add_argument("--scaler", help="default: same basename as model")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--fee",
        type=float,
        default=0.001,
        help="transaction fee per position change",
    )
    parser.add_argument("--out", help="output csv filename")
    parser.add_argument(
        "--protocol",
        type=str,
        help="Path to experiment protocol json for baseline parameters",
    )
    parser.add_argument(
        "--eval_split",
        choices=["all", "test"],
        default="test",
        help="Evaluate on full data or unseen test split",
    )
    parser.add_argument(
        "--infer_batch",
        type=int,
        default=256,
        help="Inference batch size for LSTM (avoids GPU OOM when backtesting long CSVs).",
    )
    return parser


def _interactive_args() -> list[str]:
    print("\n=== Backtest - Interactive mode ===")
    csv_in = input(f"CSV path (e.g. {SAVE_DIR.name}/AAPL_1h_2y.csv): ").strip()
    model_in = input("Model .pt path (e.g. model/AAPL/model.pt): ").strip()
    fee_in = input("fee [0.001]: ").strip() or "0.001"
    split_in = input("eval_split (test/all) [test]: ").strip().lower() or "test"
    protocol_in = input("protocol json path (blank=none): ").strip()
    threshold_in = input("threshold (blank=use meta eval_threshold): ").strip()
    args = [
        "--csv",
        csv_in.strip('"').strip("'"),
        "--model",
        model_in.strip('"').strip("'"),
        "--fee",
        fee_in,
        "--eval_split",
        split_in if split_in in {"test", "all"} else "test",
    ]
    if protocol_in:
        args += ["--protocol", protocol_in.strip('"').strip("'")]
    if threshold_in:
        args += ["--threshold", threshold_in]
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    cli_args = list(argv) if argv is not None else None
    if cli_args is None:
        import sys

        cli_args = sys.argv[1:]
    if not cli_args:
        cli_args = _interactive_args()
    args = build_parser().parse_args(cli_args)
    if not args.csv or not args.model:
        build_parser().print_help()
        raise SystemExit(
            "\n[ERROR] --csv and --model are required (or run without args for interactive mode)."
        )
    return args


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


def _write_skip_summary(
    csv_arg: str,
    reason: str,
    detail: dict,
    *,
    model_path: Path,
    meta_path: Path,
    scaler_path: Path,
) -> Path:
    symbol = infer_symbol_from_path(csv_arg)
    result_dir = RESULTS_DIR / symbol
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_path = result_dir / f"{Path(csv_arg).stem}_bt_summary.json"
    payload = {
        "status": "skipped",
        "skip_reason": reason,
        "detail": detail,
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "scaler_path": str(scaler_path),
    }
    dump_json(summary_path, payload)
    return summary_path


def run_backtest(args: argparse.Namespace) -> int:
    bundle = load_artifact_bundle(
        args.model,
        meta_path=args.meta,
        scaler_path=args.scaler,
    )
    meta = bundle.meta
    protocol = load_json(args.protocol) if args.protocol else {}

    if args.threshold is None:
        args.threshold = float(meta.get("eval_threshold", 0.5))
    if args.window == 30 and isinstance(meta.get("window"), int):
        args.window = int(meta["window"])

    df = read_backtest_frame(args.csv)

    feats = meta.get("features", [])
    use_interval_embedding = bool(meta.get("use_interval_embedding", False))
    if isinstance(meta.get("feats_lstm"), list):
        feats_lstm = meta["feats_lstm"]
    else:
        feats_lstm, _ = lstm_feature_columns(feats, use_interval_embedding)

    missing = [col for col in feats if col not in df.columns]
    if missing:
        raise ValueError(f"Missing features required by metadata: {missing}")

    scaled_feats = scaled_feature_columns(meta, feats)
    df[scaled_feats] = bundle.scaler.transform(df[scaled_feats]).astype(np.float32)
    X, iv, _, _ = build_seq_x_only(
        df,
        feats_lstm,
        args.window,
        use_interval_embedding=use_interval_embedding,
        extra_next_bar=False,
    )
    if X.shape[0] == 0:
        print("\n" + "-" * LINE_WIDTH)
        print("[SKIP] Not enough bars for this model window")
        print("-" * LINE_WIDTH)
        print(f"  csv                  : {args.csv}")
        print(f"  rows_after_fe        : {len(df)}")
        print(f"  window (effective)   : {args.window}")
        print(
            "  hint                 : need len(df) > window after indicators; "
            "low-frequency CSVs are often too short vs a large window."
        )
        summary_path = _write_skip_summary(
            args.csv,
            "insufficient_rows_after_fe",
            {"rows_after_fe": len(df), "window": args.window},
            model_path=bundle.model_path,
            meta_path=bundle.meta_path,
            scaler_path=bundle.scaler_path,
        )
        print(f"  skip_summary_path    : {summary_path}")
        print("-" * LINE_WIDTH + "\n")
        return 0

    model = build_model_from_meta(len(feats_lstm), meta, device=DEVICE)
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

    fwd_ret = df["fwd_ret"].iloc[args.window:].values.astype(np.float64)
    baseline_params = protocol.get("baseline_params", {})
    macd_p = baseline_params.get("macd", {"fast": 12, "slow": 26, "signal": 9})
    rsi_p = baseline_params.get("rsi", {"window": 14, "oversold": 30, "overbought": 70})
    bb_p = baseline_params.get("bollinger", {"window": 20, "std": 2.0})

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
    bb_high = (
        bb_obj.bollinger_hband()
        .iloc[args.window:]
        .bfill()
        .fillna(df["close"].iloc[args.window:])
        .values
    )
    bb_low = (
        bb_obj.bollinger_lband()
        .iloc[args.window:]
        .bfill()
        .fillna(df["close"].iloc[args.window:])
        .values
    )
    close_eval = df["close"].iloc[args.window:].values

    if args.eval_split == "test":
        train_ratio = float(meta.get("train_ratio", 0.7))
        val_ratio = float(meta.get("val_ratio", 0.15))
        start = int(len(fwd_ret) * (train_ratio + val_ratio))
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
    if len(fwd_ret) == 0:
        print("\n" + "-" * LINE_WIDTH)
        print("[SKIP] No rows in the chosen evaluation split")
        print("-" * LINE_WIDTH)
        print(f"  csv                  : {args.csv}")
        print(f"  eval_split           : {args.eval_split}")
        print(f"  aligned_bars         : {len(df) - args.window}")
        print(f"  window               : {args.window}")
        summary_path = _write_skip_summary(
            args.csv,
            "empty_eval_split",
            {
                "eval_split": args.eval_split,
                "aligned_bars": len(df) - args.window,
                "window": args.window,
            },
            model_path=bundle.model_path,
            meta_path=bundle.meta_path,
            scaler_path=bundle.scaler_path,
        )
        print(f"  skip_summary_path    : {summary_path}")
        print("-" * LINE_WIDTH + "\n")
        return 0

    periods_per_year = infer_periods_per_year(Path(args.csv).name)

    def strategy_metrics(signals: np.ndarray):
        turnover = np.abs(np.diff(np.insert(signals, 0, 0))).astype(np.float64)
        strat_ret = signals * fwd_ret - args.fee * turnover
        equity = np.cumprod(1.0 + strat_ret) if len(strat_ret) else np.array([], dtype=np.float64)
        roll_max = np.maximum.accumulate(equity) if len(equity) else np.array([], dtype=np.float64)
        max_dd = float(np.min(equity / roll_max - 1.0)) if len(equity) else 0.0
        sharpe = (
            float(np.sqrt(periods_per_year) * strat_ret.mean() / (strat_ret.std() + 1e-12))
            if len(strat_ret)
            else 0.0
        )
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
    signals_rsi = np.where(
        rsi_vals < float(rsi_p.get("oversold", 30)),
        1,
        np.where(rsi_vals > float(rsi_p.get("overbought", 70)), -1, 0),
    ).astype(np.int8)
    signals_bb = np.where(
        close_eval < bb_low,
        1,
        np.where(close_eval > bb_high, -1, 0),
    ).astype(np.int8)

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

    symbol = infer_symbol_from_path(args.csv)
    result_dir = RESULTS_DIR / symbol
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
        "model_path": str(bundle.model_path),
        "meta_path": str(bundle.meta_path),
        "scaler_path": str(bundle.scaler_path),
    }
    summary_path = result_dir / f"{Path(args.csv).stem}_bt_summary.json"
    dump_json(summary_path, summary)

    print("\n" + "-" * LINE_WIDTH)
    print("BACKTEST OUTPUT")
    print("-" * LINE_WIDTH)
    print(f"  backtest_output_csv_path            : {out_path}")
    print(f"  backtest_summary_json_path          : {summary_path}")

    print("\n" + "-" * LINE_WIDTH)
    print("BACKTEST METRICS")
    print("-" * LINE_WIDTH)
    print(
        "  model_lstm_total_return             : "
        f"{summary['strategies']['model_lstm']['total_return']:.4f}"
    )
    print(
        "  model_direction_accuracy            : "
        f"{summary['predictive_metrics_model']['accuracy']:.4f}"
    )
    print(
        "  model_direction_f1_score            : "
        f"{summary['predictive_metrics_model']['f1']:.4f}"
    )
    print(
        "  buy_and_hold_total_return           : "
        f"{summary['strategies']['buy_and_hold']['total_return']:.4f}"
    )
    print(
        "  macd_strategy_total_return          : "
        f"{summary['strategies']['macd']['total_return']:.4f}"
    )
    print(
        "  rsi_strategy_total_return           : "
        f"{summary['strategies']['rsi']['total_return']:.4f}"
    )
    print(
        "  bollinger_band_strategy_total_return: "
        f"{summary['strategies']['bollinger']['total_return']:.4f}"
    )
    print("-" * LINE_WIDTH)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_backtest(args)


if __name__ == "__main__":
    raise SystemExit(main())
