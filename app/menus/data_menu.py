import subprocess

from ..actions import run_script
from ..constants import INTERVAL_LIMITS
from ..services import duration_to_years, max_lookback_label
from ..ui import ask, ask_yes_no, clear_screen, pause


def _print_supported_intervals() -> None:
    for iv, days in INTERVAL_LIMITS.items():
        print(f"  - {iv}/{max_lookback_label(days)}")
    print("  all = every interval above\n")


def _parse_intervals(interval_raw: str) -> tuple[list[str] | None, str | None]:
    tokens = [x.strip() for x in interval_raw.lower().replace(",", " ").split() if x.strip()]
    if not tokens:
        return None, "interval(s) cannot be empty."
    if "all" in tokens:
        if len(tokens) != 1:
            return None, "Use 'all' alone to select every supported interval."
        return list(INTERVAL_LIMITS.keys()), None
    intervals = tokens
    if any(interval not in INTERVAL_LIMITS for interval in intervals):
        return None, f"Unsupported interval(s): {interval_raw}"
    return intervals, None


def _collect_download_config(
    ticker_scope: str,
    *,
    default_intervals: str = "all",
    default_range: str = "max",
) -> tuple[list[str] | None, str | None]:
    print(f"\n  Configure: {ticker_scope}")
    _print_supported_intervals()
    interval_raw = ask("intervals (e.g. 1d,1wk or all)", default_intervals).lower()
    intervals, interval_error = _parse_intervals(interval_raw)
    if interval_error:
        print(f"\n  [ERROR] {interval_error}")
        if "Unsupported interval" in interval_error:
            print(f"  Supported: {', '.join(INTERVAL_LIMITS.keys())}, or 'all'")
        return None, None
    assert intervals is not None
    duration_raw = ask("range (e.g. 10y,60d,max)", default_range)
    years_val = duration_to_years(duration_raw)
    if years_val is None or years_val <= 0:
        print("\n  [ERROR] Invalid range. Try 30d, 6mo, 2y, or max.")
        return None, None
    years_arg = f"{years_val:g}"
    return intervals, years_arg


def _print_data_menu():
    print("\n" + "=" * 70)
    print("Stock ML - Data Menu")
    print("=" * 70)
    print("\nPlease select an option:\n")
    print("[Download]")
    print("   1. Download by ticker/interval(s)/historical range (get_stock_data.py)")
    print("\n   0. Back")
    print("=" * 70)


def run_data_menu():
    while True:
        clear_screen()
        _print_data_menu()
        try:
            choice = input("\nEnter your choice [1]: ").strip() or "1"
            print("")
        except (KeyboardInterrupt, EOFError):
            print("\nBack")
            break
        if choice == "0":
            break
        if choice == "1":
            tickers_raw = ask("tickers (e.g. AAPL,MSFT)", "AAPL")
            tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
            if not tickers:
                print("\n  [ERROR] tickers cannot be empty.")
                pause()
                continue
            print(f"  tickers: {', '.join(tickers)}\n")
            use_same_config = True if len(tickers) == 1 else ask_yes_no("Use same intervals/range for all tickers?", default=True)
            jobs: list[tuple[str, list[str], str]] = []
            if use_same_config:
                intervals, years_arg = _collect_download_config(", ".join(tickers))
                if intervals is None or years_arg is None:
                    pause()
                    continue
                jobs = [(ticker, intervals, years_arg) for ticker in tickers]
            else:
                for ticker in tickers:
                    intervals, years_arg = _collect_download_config(ticker)
                    if intervals is None or years_arg is None:
                        jobs = []
                        break
                    jobs.append((ticker, intervals, years_arg))
                if not jobs:
                    pause()
                    continue

            print("\n  Download plan:")
            for ticker, intervals, years_arg in jobs:
                interval_text = ",".join(intervals)
                full_interval_set = set(intervals) == set(INTERVAL_LIMITS.keys())
                interval_display = f"all ({len(intervals)})" if full_interval_set else interval_text
                interval_limits = ", ".join(f"{i}->{max_lookback_label(INTERVAL_LIMITS[i])}" for i in intervals)
                caps_note = "per-interval Yahoo caps" if full_interval_set else interval_limits
                print(f"  - {ticker} | {interval_display} | {years_arg}y ({caps_note})")
            if ask_yes_no("Ready to download?", default=True):
                print("")
                try:
                    for ticker, intervals, years_arg in jobs:
                        run_script(
                            "get_stock_data.py",
                            ["--ticker", ticker, "--interval", *intervals, "--years", years_arg],
                        )
                except subprocess.CalledProcessError as exc:
                    print(f"\n  [ERROR] Download failed (exit {exc.returncode}).")
                    pause()
                    continue
                pause()
                break
            else:
                print("  Cancelled.")
                pause()
        else:
            print("\nInvalid choice, please try again")

