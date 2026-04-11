import os, argparse, datetime, sys, json, warnings
from pathlib import Path
import pandas as pd
import yfinance as yf

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.paths import SAVE_DIR


def ensure_project_folders() -> None:
    """Create download folder from SAVE_DIR (.env)."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)


ensure_project_folders()

# ---------- Parse arguments ----------
# Interval -> max lookback days (used for prompts and validation)
INTERVAL_LIMITS = {
    "1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60,
    "60m": 730, "90m": 60, "1h": 730,
    "1d": 3650, "5d": 3650, "1wk": 3650, "1mo": 3650, "3mo": 3650
}
INTERVAL_OPTIONS = list(INTERVAL_LIMITS.keys())

def _max_lookback_label(max_days: int) -> str:
    """Human-friendly max lookback, e.g. 730 -> '2y', 7 -> '7d'."""
    return f"{round(max_days / 365)}y" if max_days >= 365 else f"{max_days}d"

def _validate_tickers(tickers: list[str]) -> list[str]:
    """Return list of ticker(s) that are invalid or have no data."""
    invalid = []
    for tic in tickers:
        try:
            df = yf.Ticker(tic.strip().upper()).history(period="5d")
            if df is None or df.empty:
                invalid.append(tic.strip().upper())
        except Exception:
            invalid.append(tic.strip().upper())
    return invalid

def get_args():
    """
    Non-interactive CLI.
    - Either provide --protocol (and optional --window_idx)
    - Or provide --ticker + --interval + --years
    """
    p = argparse.ArgumentParser("Stock OHLCV Downloader (menu-driven)")
    p.add_argument("--protocol", type=str, help="Path to experiment protocol json")
    p.add_argument("--window_idx", type=int, default=0, help="Index of data window from protocol")

    p.add_argument("--ticker", nargs="+", help="Stock ticker(s), space-separated for multiple")
    p.add_argument("--years", type=float, help="Lookback years (float, e.g. 2, 0.5)")
    p.add_argument("--interval", nargs="+", help="Data interval(s): " + ", ".join(INTERVAL_OPTIONS))

    a = p.parse_args()

    # Interactive fallback when launched directly without args.
    if len(sys.argv) == 1:
        print("\n=== Download ‧ Interactive mode ===")
        t = input("Enter stock ticker(s), space-separated: ").strip()
        if not t:
            print("No ticker entered.")
            sys.exit(0)
        a.ticker = [x.strip().upper() for x in t.split()]

        opts = ", ".join(INTERVAL_OPTIONS)
        while True:
            interval_in = input(f"Enter data interval(s), comma-separated (default 1d). Options: {opts}\n> ").strip().lower() or "1d"
            interval_items = [x.strip() for x in interval_in.replace(",", " ").split() if x.strip()]
            if interval_items and all(item in INTERVAL_LIMITS for item in interval_items):
                a.interval = interval_items
                break
            print(f"Unsupported interval(s) '{interval_in}'.")

        max_days = max(INTERVAL_LIMITS[item] for item in a.interval)
        max_label = _max_lookback_label(max_days)
        dur = input(f"Enter historical range (e.g. 30d, 6mo, 2y), max {max_label} [{max_label}]: ").strip() or max_label
        dur_low = dur.lower()
        if dur_low.endswith("mo") and dur_low[:-2].strip().replace(".", "", 1).isdigit():
            a.years = float(dur_low[:-2].strip()) / 12
        elif dur_low.endswith("d") and dur_low[:-1].strip().replace(".", "", 1).isdigit():
            a.years = float(dur_low[:-1].strip()) / 365
        elif dur_low.endswith("y") and dur_low[:-1].strip().replace(".", "", 1).isdigit():
            a.years = float(dur_low[:-1].strip())
        elif dur_low.replace(".", "", 1).isdigit():
            a.years = float(dur_low)
        else:
            print("Invalid duration format. Use 30d / 6mo / 2y.")
            sys.exit(2)

    if a.protocol:
        try:
            with open(a.protocol, "r", encoding="utf-8") as f:
                protocol = json.load(f)
        except Exception as e:
            print(f"Failed to read protocol {a.protocol}: {e}")
            sys.exit(1)
        tickers = protocol.get("ticker_universe", [])
        windows = protocol.get("data_windows", [])
        if not tickers or not windows:
            print("Protocol must include non-empty ticker_universe and data_windows.")
            sys.exit(1)
        if a.window_idx < 0 or a.window_idx >= len(windows):
            print(f"window_idx out of range [0, {len(windows)-1}]")
            sys.exit(1)
        w = windows[a.window_idx]
        a.ticker = [str(t).strip().upper() for t in tickers]
        a.interval = [str(w.get("interval", "1d")).lower()]
        a.years = float(w.get("years", 5))
        return a

    # Non-protocol mode must be fully specified
    if not a.ticker or a.years is None or not a.interval:
        p.print_help()
        print("\n[ERROR] Missing required args. Use --protocol ... OR --ticker ... --interval ... --years ...")
        sys.exit(2)

    a.ticker = [x.strip().upper() for x in a.ticker]
    a.interval = [x.lower() for x in a.interval]
    invalid = _validate_tickers(a.ticker)
    if invalid:
        print(f"Invalid or no data for: {', '.join(invalid)}.")
        sys.exit(1)
    return a

args = get_args()

# Normalize interval to lowercase (CLI e.g. --interval 5D -> 5d)
if args.interval is not None:
    args.interval = [x.lower() for x in args.interval]

# ---------- Interval supported range ----------
for interval in args.interval:
    if interval not in INTERVAL_LIMITS:
        print(f"Unsupported interval '{interval}'. Available: {', '.join(INTERVAL_OPTIONS)}")
        sys.exit(1)

# ---------- Auto-cap lookback days per interval ----------
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

# ---------- Main loop ----------
for tic in args.ticker:
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

        print(f"\nDownloading {tic}, last {lookback_label}, interval {interval}…")
        try:
            # yfinance compares Timedelta(interval); pandas deprecates lowercase 'd' in some paths.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*'d' is deprecated.*use 'D' instead of 'd'.*",
                )
                df = yf.download(
                    tic,
                    start=start_date,
                    interval=interval,
                    auto_adjust=True,
                    progress=False
                )
        except Exception as e:
            print(f"Download failed for {tic} {interval}: {e}")
            continue

        if df.empty:
            print(f"No data for {tic} {interval}; skipping.")
            continue

        # ---------- Reset and flatten columns ----------
        df = df.reset_index(drop=False)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.droplevel(1)
            except (IndexError, KeyError):
                df.columns = ["_".join(filter(None, map(str, col))).lower() for col in df.columns]
            else:
                df.columns = [str(c).lower() for c in df.columns]
        else:
            df.columns = [str(col).lower() for col in df.columns]

        # ---------- Detect time column ----------
        if "datetime" in df.columns:
            df.rename(columns={"datetime": "date"}, inplace=True)

        if "date" not in df.columns:
            print("No date/datetime column found; skipping.")
            print("DataFrame columns:", df.columns.tolist())
            print("Column dtypes:\n", df.dtypes)
            print("First 5 rows:\n", df.head())
            continue

        # ---------- Normalize OHLCV column names ----------
        for base in ["open", "high", "low", "close", "volume"]:
            match = next((c for c in df.columns if c.startswith(base)), None)
            if match:
                df.rename(columns={match: base}, inplace=True)

        # ---------- Required columns check ----------
        needed = ["date", "open", "high", "low", "close", "volume"]
        if not set(needed).issubset(df.columns):
            print(f"Missing columns {set(needed) - set(df.columns)}; skipping.")
            continue
        df = df[needed]

        # ---------- Write CSV ----------
        fname = f"{tic.upper()}_{interval}_{lookback_label}.csv"
        outpath = str(SAVE_DIR / fname)
        df.to_csv(outpath, index=False)
        out_path_obj = Path(outpath).resolve()
        try:
            display_path = out_path_obj.relative_to(_PROJECT_ROOT.resolve())
        except ValueError:
            display_path = out_path_obj
        print(f"Saved {display_path}")
