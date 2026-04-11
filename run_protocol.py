#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_protocol.py -- batch runner for protocol-based experiments.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from app.paths import SAVE_DIR, RESULTS_DIR, MODEL_DIR

import pandas as pd


def lookback_label(years: float) -> str:
    days = max(1, round(years * 365))
    if days <= 90:
        return f"{days}d"
    if days < 365:
        return f"{round(days / 30)}mo"
    return f"{round(days / 365)}y"


def run_cmd(cmd: list[str]):
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


ap = argparse.ArgumentParser()
ap.add_argument("--protocol", help="Path to experiment protocol json")
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--window", type=int, default=30)
ap.add_argument("--fee", type=float, default=0.001)
ap.add_argument("--tickers", nargs="*", help="Optional ticker subset")
ap.add_argument("--window_idxs", nargs="*", type=int, help="Optional data window index subset")
ap.add_argument("--out", help="Output comparison csv path")
args = ap.parse_args()

if len(sys.argv) == 1:
    print("\n=== Protocol Runner ‧ Interactive mode ===")
    protocol_in = input("protocol json path [experiment_protocol.json]: ").strip() or "experiment_protocol.json"
    epochs_in = input("epochs [5]: ").strip() or "5"
    window_in = input("window [30]: ").strip() or "30"
    fee_in = input("fee [0.001]: ").strip() or "0.001"
    _def_cmp = str(RESULTS_DIR / "comparison_summary.csv")
    out_in = input(f"out csv path (blank=default {_def_cmp}): ").strip()
    tickers_in = input("ticker subset (space-separated, blank=all from protocol): ").strip()
    idxs_in = input("window_idx subset (space-separated ints, blank=all): ").strip()

    args.protocol = protocol_in.strip('"').strip("'")
    args.epochs = int(epochs_in)
    args.window = int(window_in)
    args.fee = float(fee_in)
    args.out = out_in.strip('"').strip("'") or None
    args.tickers = tickers_in.split() if tickers_in else None
    args.window_idxs = [int(x) for x in idxs_in.split()] if idxs_in else None

if not args.protocol:
    ap.print_help()
    raise SystemExit("\n[ERROR] --protocol is required (or run without args for interactive mode).")

protocol_path = Path(args.protocol)
with open(protocol_path, "r", encoding="utf-8") as f:
    protocol = json.load(f)

all_tickers = [str(t).upper() for t in protocol.get("ticker_universe", [])]
if not all_tickers:
    raise ValueError("Protocol ticker_universe is empty")
tickers = [t.upper() for t in args.tickers] if args.tickers else all_tickers

windows = protocol.get("data_windows", [])
if not windows:
    raise ValueError("Protocol data_windows is empty")
idxs = args.window_idxs if args.window_idxs else list(range(len(windows)))

rows = []
for widx in idxs:
    if widx < 0 or widx >= len(windows):
        raise ValueError(f"window_idx out of range: {widx}")
    w = windows[widx]
    interval = str(w.get("interval", "1d")).lower()
    years = float(w.get("years", 5))

    # Step 1: fetch data for protocol window
    run_cmd(
        [
            sys.executable,
            "get_stock_data.py",
            "--protocol",
            str(protocol_path),
            "--window_idx",
            str(widx),
        ]
    )

    lb = lookback_label(years)
    for tic in tickers:
        csv_path = SAVE_DIR / f"{tic}_{interval}_{lb}.csv"
        if not csv_path.exists():
            print(f"[WARN] skip {tic}: csv not found {csv_path}")
            continue

        # Step 2: train model for each ticker using protocol split settings
        run_cmd(
            [
                sys.executable,
                "train_stock.py",
                "--csv_dir",
                str(SAVE_DIR),
                "--ticker",
                tic,
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

        # Step 3: backtest and compare against baselines on test split
        run_cmd(
            [
                sys.executable,
                "backtest_stock.py",
                "--csv",
                str(csv_path),
                "--model",
                str(MODEL_DIR / tic / "model.pt"),
                "--protocol",
                str(protocol_path),
                "--eval_split",
                "test",
                "--fee",
                str(args.fee),
            ]
        )

        summary_path = RESULTS_DIR / tic / f"{csv_path.stem}_bt_summary.json"
        with open(summary_path, "r", encoding="utf-8") as f:
            s = json.load(f)
        for strat, vals in s.get("strategies", {}).items():
            pred = s.get("predictive_metrics_model", {})
            rows.append(
                {
                    "protocol": protocol.get("name", "unnamed"),
                    "window_idx": widx,
                    "interval": interval,
                    "years": years,
                    "ticker": tic,
                    "strategy": strat,
                    "total_return": vals.get("total_return"),
                    "sharpe": vals.get("sharpe"),
                    "max_drawdown": vals.get("max_drawdown"),
                    "win_rate": vals.get("win_rate"),
                    "avg_turnover": vals.get("avg_turnover"),
                    "turnover_rate": vals.get("turnover_rate"),
                    "n_trades": vals.get("n_trades"),
                    "model_accuracy": pred.get("accuracy") if strat == "model_lstm" else None,
                    "model_f1": pred.get("f1") if strat == "model_lstm" else None,
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
