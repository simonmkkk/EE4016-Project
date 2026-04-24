from __future__ import annotations

import argparse
import datetime
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

from ..constants import INTERVAL_LIMITS
from ..infra.files import ensure_parent
from ..paths import SAVE_DIR


INTERVAL_OPTIONS = list(INTERVAL_LIMITS.keys())


def ensure_project_folders() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)


def _max_lookback_label(max_days: int) -> str:
    return f"{round(max_days / 365)}y" if max_days >= 365 else f"{max_days}d"


def _validate_tickers(tickers: list[str]) -> list[str]:
    invalid = []
    for ticker in tickers:
        try:
            df = yf.Ticker(ticker.strip().upper()).history(period="5d")
            if df is None or df.empty:
                invalid.append(ticker.strip().upper())
        except Exception:
            invalid.append(ticker.strip().upper())
    return invalid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Stock OHLCV Downloader (menu-driven)")
    parser.add_argument("--protocol", type=str, help="Path to experiment protocol json")
    parser.add_argument(
        "--window_idx",
        type=int,
        default=0,
        help="Index of data window from protocol",
    )
    parser.add_argument(
        "--ticker",
        nargs="+",
        help="Stock ticker(s), space-separated for multiple",
    )
    parser.add_argument("--years", type=float, help="Lookback years (float, e.g. 2, 0.5)")
    parser.add_argument(
        "--interval",
        nargs="+",
        help="Data interval(s): " + ", ".join(INTERVAL_OPTIONS),
    )
    return parser


def _interactive_args() -> list[str]:
    print("\n=== Download - Interactive mode ===")
    tickers = input("Enter stock ticker(s), space-separated: ").strip()
    if not tickers:
        raise SystemExit(0)

    options = ", ".join(INTERVAL_OPTIONS)
    while True:
        interval_in = (
            input(
                "Enter data interval(s), comma-separated (default 1d). "
                f"Options: {options}\n> "
            ).strip().lower()
            or "1d"
        )
        interval_items = [
            item.strip()
            for item in interval_in.replace(",", " ").split()
            if item.strip()
        ]
        if interval_items and all(item in INTERVAL_LIMITS for item in interval_items):
            break
        print(f"Unsupported interval(s) '{interval_in}'.")

    max_days = max(INTERVAL_LIMITS[item] for item in interval_items)
    max_label = _max_lookback_label(max_days)
    duration = (
        input(
            f"Enter historical range (e.g. 30d, 6mo, 2y), max {max_label} "
            f"[{max_label}]: "
        ).strip()
        or max_label
    )
    duration_low = duration.lower()
    if duration_low.endswith("mo") and duration_low[:-2].strip().replace(".", "", 1).isdigit():
        years = float(duration_low[:-2].strip()) / 12
    elif duration_low.endswith("d") and duration_low[:-1].strip().replace(".", "", 1).isdigit():
        years = float(duration_low[:-1].strip()) / 365
    elif duration_low.endswith("y") and duration_low[:-1].strip().replace(".", "", 1).isdigit():
        years = float(duration_low[:-1].strip())
    elif duration_low.replace(".", "", 1).isdigit():
        years = float(duration_low)
    else:
        raise SystemExit("Invalid duration format. Use 30d / 6mo / 2y.")

    return [
        "--ticker",
        *[item.strip().upper() for item in tickers.split()],
        "--interval",
        *interval_items,
        "--years",
        str(years),
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    cli_args = list(argv) if argv is not None else None
    if cli_args is None:
        import sys

        cli_args = sys.argv[1:]
    if not cli_args:
        cli_args = _interactive_args()
    args = parser.parse_args(cli_args)

    if args.protocol:
        from ..infra.files import load_json

        protocol = load_json(args.protocol)
        tickers = protocol.get("ticker_universe", [])
        windows = protocol.get("data_windows", [])
        if not tickers or not windows:
            raise SystemExit(
                "Protocol must include non-empty ticker_universe and data_windows."
            )
        if args.window_idx < 0 or args.window_idx >= len(windows):
            raise SystemExit(f"window_idx out of range [0, {len(windows) - 1}]")
        window = windows[args.window_idx]
        args.ticker = [str(item).strip().upper() for item in tickers]
        args.interval = [str(window.get("interval", "1d")).lower()]
        args.years = float(window.get("years", 5))
        return args

    if not args.ticker or args.years is None or not args.interval:
        parser.print_help()
        raise SystemExit(
            "\n[ERROR] Missing required args. Use --protocol ... OR "
            "--ticker ... --interval ... --years ..."
        )

    args.ticker = [item.strip().upper() for item in args.ticker]
    args.interval = [item.lower() for item in args.interval]
    invalid = _validate_tickers(args.ticker)
    if invalid:
        raise SystemExit(f"Invalid or no data for: {', '.join(invalid)}.")
    return args


def run_download(args: argparse.Namespace) -> int:
    ensure_project_folders()

    for interval in args.interval:
        if interval not in INTERVAL_LIMITS:
            raise SystemExit(
                f"Unsupported interval '{interval}'. Available: {', '.join(INTERVAL_OPTIONS)}"
            )

    interval_years = []
    for interval in args.interval:
        max_days = INTERVAL_LIMITS[interval]
        desired_days = args.years * 365
        if desired_days >= max_days:
            adjusted_days = max(1, max_days - 1)
            print(
                f"{interval} supports at most {_max_lookback_label(max_days)}; "
                f"auto-adjust to {adjusted_days}d to avoid yfinance boundary issue."
            )
            interval_years.append(adjusted_days / 365)
        elif desired_days > max_days:
            print(f"{interval} supports at most {_max_lookback_label(max_days)}; adjusted.")
            interval_years.append(max_days / 365)
        else:
            interval_years.append(args.years)

    if args.years <= 0:
        args.years = (1 / 365) if any(i.endswith("m") or i == "1h" for i in args.interval) else 1
        print("Lookback too small; set to minimum.")
        interval_years = [args.years] * len(args.interval)

    for ticker in args.ticker:
        for interval, interval_year in zip(args.interval, interval_years):
            max_days = INTERVAL_LIMITS[interval]
            lookback_days = max(1, round(interval_year * 365))
            if lookback_days >= max_days:
                lookback_days = max(1, max_days - 1)
            start_date = datetime.date.today() - datetime.timedelta(days=lookback_days)

            if lookback_days <= 90:
                lookback_label = f"{lookback_days}d"
            elif lookback_days < 365:
                lookback_label = f"{round(lookback_days / 30)}mo"
            else:
                lookback_label = f"{round(lookback_days / 365)}y"

            print(f"\nDownloading {ticker}, last {lookback_label}, interval {interval}...")
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r".*'d' is deprecated.*use 'D' instead of 'd'.*",
                    )
                    df = yf.download(
                        ticker,
                        start=start_date,
                        interval=interval,
                        auto_adjust=True,
                        progress=False,
                    )
            except Exception as exc:
                print(f"Download failed for {ticker} {interval}: {exc}")
                continue

            if df.empty:
                print(f"No data for {ticker} {interval}; skipping.")
                continue

            df = df.reset_index(drop=False)
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df.columns = df.columns.droplevel(1)
                except (IndexError, KeyError):
                    df.columns = [
                        "_".join(filter(None, map(str, col))).lower()
                        for col in df.columns
                    ]
                else:
                    df.columns = [str(col).lower() for col in df.columns]
            else:
                df.columns = [str(col).lower() for col in df.columns]

            if "datetime" in df.columns:
                df.rename(columns={"datetime": "date"}, inplace=True)

            if "date" not in df.columns:
                print("No date/datetime column found; skipping.")
                print("DataFrame columns:", df.columns.tolist())
                print("Column dtypes:\n", df.dtypes)
                print("First 5 rows:\n", df.head())
                continue

            for base in ["open", "high", "low", "close", "volume"]:
                match = next((col for col in df.columns if col.startswith(base)), None)
                if match:
                    df.rename(columns={match: base}, inplace=True)

            needed = ["date", "open", "high", "low", "close", "volume"]
            if not set(needed).issubset(df.columns):
                print(f"Missing columns {set(needed) - set(df.columns)}; skipping.")
                continue

            out_path = ensure_parent(SAVE_DIR / f"{ticker.upper()}_{interval}_{lookback_label}.csv")
            df[needed].to_csv(out_path, index=False)
            print(f"Saved {out_path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_download(args)


if __name__ == "__main__":
    raise SystemExit(main())
