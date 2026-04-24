from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..infra.files import load_json
from ..paths import RESULTS_DIR, SAVE_DIR, MODEL_DIR
from .backtest_stock import main as backtest_main
from .get_stock_data import main as download_main
from .train_stock import main as train_main


def lookback_label(years: float) -> str:
    days = max(1, round(years * 365))
    if days <= 90:
        return f"{days}d"
    if days < 365:
        return f"{round(days / 30)}mo"
    return f"{round(days / 365)}y"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", help="Path to experiment protocol json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--tickers", nargs="*", help="Optional ticker subset")
    parser.add_argument("--window_idxs", nargs="*", type=int, help="Optional data window index subset")
    parser.add_argument("--out", help="Output comparison csv path")
    return parser


def _interactive_args() -> list[str]:
    print("\n=== Protocol Runner - Interactive mode ===")
    protocol_in = input("protocol json path [experiment_protocol.json]: ").strip() or "experiment_protocol.json"
    epochs_in = input("epochs [5]: ").strip() or "5"
    window_in = input("window [30]: ").strip() or "30"
    fee_in = input("fee [0.001]: ").strip() or "0.001"
    default_out = str(RESULTS_DIR / "comparison_summary.csv")
    out_in = input(f"out csv path (blank=default {default_out}): ").strip()
    tickers_in = input("ticker subset (space-separated, blank=all from protocol): ").strip()
    idxs_in = input("window_idx subset (space-separated ints, blank=all): ").strip()
    args = [
        "--protocol",
        protocol_in.strip('"').strip("'"),
        "--epochs",
        epochs_in,
        "--window",
        window_in,
        "--fee",
        fee_in,
    ]
    if out_in:
        args += ["--out", out_in.strip('"').strip("'")]
    if tickers_in:
        args += ["--tickers", *tickers_in.split()]
    if idxs_in:
        args += ["--window_idxs", *idxs_in.split()]
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    cli_args = list(argv) if argv is not None else None
    if cli_args is None:
        import sys

        cli_args = sys.argv[1:]
    if not cli_args:
        cli_args = _interactive_args()
    args = build_parser().parse_args(cli_args)
    if not args.protocol:
        build_parser().print_help()
        raise SystemExit("\n[ERROR] --protocol is required (or run without args for interactive mode).")
    return args


def run_protocol(args: argparse.Namespace) -> int:
    protocol_path = Path(args.protocol)
    protocol = load_json(protocol_path)

    all_tickers = [str(item).upper() for item in protocol.get("ticker_universe", [])]
    if not all_tickers:
        raise ValueError("Protocol ticker_universe is empty")
    tickers = [ticker.upper() for ticker in args.tickers] if args.tickers else all_tickers

    windows = protocol.get("data_windows", [])
    if not windows:
        raise ValueError("Protocol data_windows is empty")
    idxs = args.window_idxs if args.window_idxs else list(range(len(windows)))

    rows: list[dict] = []
    for window_idx in idxs:
        if window_idx < 0 or window_idx >= len(windows):
            raise ValueError(f"window_idx out of range: {window_idx}")
        window = windows[window_idx]
        interval = str(window.get("interval", "1d")).lower()
        years = float(window.get("years", 5))

        download_main(
            [
                "--protocol",
                str(protocol_path),
                "--window_idx",
                str(window_idx),
            ]
        )

        lb = lookback_label(years)
        for ticker in tickers:
            csv_path = SAVE_DIR / f"{ticker}_{interval}_{lb}.csv"
            if not csv_path.exists():
                print(f"[WARN] skip {ticker}: csv not found {csv_path}")
                continue

            train_main(
                [
                    "--csv_dir",
                    str(SAVE_DIR),
                    "--ticker",
                    ticker,
                    "--save_model",
                    "model.pt",
                    "--window",
                    str(args.window),
                    "--epochs",
                    str(args.epochs),
                    "--protocol",
                    str(protocol_path),
                ]
            )

            backtest_main(
                [
                    "--csv",
                    str(csv_path),
                    "--model",
                    str(MODEL_DIR / ticker / "model.pt"),
                    "--protocol",
                    str(protocol_path),
                    "--eval_split",
                    "test",
                    "--fee",
                    str(args.fee),
                ]
            )

            summary_path = RESULTS_DIR / ticker / f"{csv_path.stem}_bt_summary.json"
            summary = load_json(summary_path)
            for strategy, values in summary.get("strategies", {}).items():
                predictive = summary.get("predictive_metrics_model", {})
                rows.append(
                    {
                        "protocol": protocol.get("name", "unnamed"),
                        "window_idx": window_idx,
                        "interval": interval,
                        "years": years,
                        "ticker": ticker,
                        "strategy": strategy,
                        "total_return": values.get("total_return"),
                        "sharpe": values.get("sharpe"),
                        "max_drawdown": values.get("max_drawdown"),
                        "win_rate": values.get("win_rate"),
                        "avg_turnover": values.get("avg_turnover"),
                        "turnover_rate": values.get("turnover_rate"),
                        "n_trades": values.get("n_trades"),
                        "model_accuracy": predictive.get("accuracy") if strategy == "model_lstm" else None,
                        "model_f1": predictive.get("f1") if strategy == "model_lstm" else None,
                    }
                )

    if not rows:
        raise RuntimeError("No results collected. Check data/training/backtest outputs.")

    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["protocol", "window_idx", "interval", "years", "strategy"], as_index=False)
        .agg(
            tickers=("ticker", "nunique"),
            avg_total_return=("total_return", "mean"),
            avg_sharpe=("sharpe", "mean"),
            avg_max_drawdown=("max_drawdown", "mean"),
            avg_win_rate=("win_rate", "mean"),
            avg_turnover=("avg_turnover", "mean"),
            avg_turnover_rate=("turnover_rate", "mean"),
            avg_n_trades=("n_trades", "mean"),
            avg_model_accuracy=("model_accuracy", "mean"),
            avg_model_f1=("model_f1", "mean"),
        )
        .sort_values(["window_idx", "avg_sharpe"], ascending=[True, False])
    )

    out_path = Path(args.out) if args.out else RESULTS_DIR / "comparison_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] comparison summary saved to {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_protocol(args)


if __name__ == "__main__":
    raise SystemExit(main())
