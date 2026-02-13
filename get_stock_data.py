import os, argparse, datetime, sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import yfinance as yf


load_dotenv()
SAVE_DIR = os.getenv("SAVE_DIR", str(Path(__file__).resolve().parent / "record"))
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- Parse arguments ----------
# Interval -> max lookback days (used for prompts and validation)
INTERVAL_LIMITS = {
    "1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60,
    "60m": 730, "90m": 730, "1h": 730,
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
    p = argparse.ArgumentParser("Stock OHLCV Downloader (interactive-friendly)")
    p.add_argument("--ticker", nargs="+", help="Stock ticker(s), space-separated for multiple")
    p.add_argument("--years", type=float, help="Lookback years (float, e.g. 0.x)")
    p.add_argument("--interval", type=str, help="Data interval: " + ", ".join(INTERVAL_OPTIONS))
    a = p.parse_args()

    # Tickers: always normalized to uppercase (e.g. aapl -> AAPL)
    if not a.ticker:
        while True:
            t = input("Enter stock ticker(s): ").strip()
            if not t:
                print("No ticker entered.")
                sys.exit(0)
            a.ticker = [x.strip().upper() for x in t.split()]
            invalid = _validate_tickers(a.ticker)
            if not invalid:
                break
            print(f"Invalid or no data for: {', '.join(invalid)}. Please re-enter ticker(s).\n")
    else:
        a.ticker = [x.strip().upper() for x in a.ticker]
        invalid = _validate_tickers(a.ticker)
        if invalid:
            print(f"Invalid or no data for: {', '.join(invalid)}.")
            sys.exit(1)

    # Interval: normalized to lowercase (e.g. 5D -> 5d)
    if a.interval is None:
        opts = ", ".join(INTERVAL_OPTIONS)
        while True:
            it = input(f"Enter data interval (default 1d). Options: {opts}\n> ").strip()
            a.interval = (it or "1d").lower()
            if a.interval in INTERVAL_LIMITS:
                break
            print(f"Unsupported interval '{a.interval}'. Available: {opts}\n")

    if a.years is None:
        hint = "default 5 years"
        if a.interval in INTERVAL_LIMITS:
            md = INTERVAL_LIMITS[a.interval]
            hint = f"default 5 years, max {_max_lookback_label(md)}"
        raw = input(f"Enter lookback: endwith [d/mo/y] ({hint}): ").strip()
        if not raw:
            a.years = 5
        elif raw.endswith("d") and len(raw) > 1 and raw[:-1].replace(".", "", 1).isdigit():
            a.years = float(raw[:-1]) / 365
        elif raw.endswith("mo") and raw[:-2].strip().replace(".", "", 1).isdigit():
            a.years = float(raw[:-2].strip()) / 12
        elif raw.endswith("y") and len(raw) > 1 and raw[:-1].replace(".", "", 1).isdigit():
            a.years = float(raw[:-1])
        elif raw.replace(".", "", 1).isdigit() and float(raw) > 0:
            a.years = float(raw)
        else:
            a.years = 5

    return a

args = get_args()

# Normalize interval to lowercase (CLI e.g. --interval 5D -> 5d)
if args.interval is not None:
    args.interval = args.interval.lower()

# ---------- Interval supported range ----------
if args.interval not in INTERVAL_LIMITS:
    print(f"Unsupported interval '{args.interval}'. Available: {', '.join(INTERVAL_OPTIONS)}")
    sys.exit(1)

# ---------- Auto-cap lookback days ----------
max_days = INTERVAL_LIMITS[args.interval]
max_years = max_days / 365
if args.years * 365 > max_days:
    print(f"{args.interval} supports at most {_max_lookback_label(max_days)}; adjusted.")
    args.years = max_years

if args.years <= 0:
    args.years = (1 / 365) if args.interval.endswith("m") or args.interval == "1h" else 1
    print("Lookback too small; set to minimum.")

lookback_days = max(1, round(args.years * 365))
start_date = datetime.date.today() - datetime.timedelta(days=lookback_days)
# Do not pass end= to yf.download: API treats end as exclusive, so we'd miss today's data
# Label: days (≤90), months (<1y), or years
if lookback_days <= 90:
    lookback_label = f"{lookback_days}d"
elif lookback_days < 365:
    lookback_label = f"{round(lookback_days / 30)}mo"
else:
    lookback_label = f"{round(lookback_days / 365)}y"

# ---------- Main loop ----------
for tic in args.ticker:
    print(f"\nDownloading {tic}, last {lookback_label}, interval {args.interval}…")
    try:
        df = yf.download(
            tic,
            start=start_date,
            interval=args.interval,
            auto_adjust=True,
            progress=False
        )
    except Exception as e:
        print(f"Download failed for {tic}: {e}")
        continue

    if df.empty:
        print(f"No data for {tic}; skipping.")
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
    fname = f"{tic.upper()}_{lookback_label}_{args.interval}.csv"
    outpath = os.path.join(SAVE_DIR, fname)
    df.to_csv(outpath, index=False)
    print(f"Saved {outpath}")
