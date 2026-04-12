from ..actions import run_script
from ..constants import INTERVAL_LIMITS
from ..services import duration_to_years, max_lookback_label
from ..ui import ask, ask_yes_no, clear_screen, pause


def _print_supported_intervals() -> None:
    caps = " ".join(f"{iv}/{max_lookback_label(d)}" for iv, d in INTERVAL_LIMITS.items())
    print(f"  {caps}")
    print("  all = every interval above\n")


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
            _print_supported_intervals()
            interval_raw = ask("intervals (comma-sep or all)", "1d,1h").lower()
            tokens = [x.strip() for x in interval_raw.replace(",", " ").split() if x.strip()]
            if not tokens:
                print("\n  [ERROR] interval(s) cannot be empty.")
                pause()
                continue
            if "all" in tokens:
                if len(tokens) != 1:
                    print("\n  [ERROR] Use 'all' alone to select every supported interval.")
                    pause()
                    continue
                intervals = list(INTERVAL_LIMITS.keys())
            else:
                intervals = tokens
                if any(interval not in INTERVAL_LIMITS for interval in intervals):
                    print(f"\n  [ERROR] Unsupported interval(s): {interval_raw}")
                    print(f"  Supported: {', '.join(INTERVAL_LIMITS.keys())}, or 'all'")
                    pause()
                    continue
            interval = ",".join(intervals)
            interval_max_days = [INTERVAL_LIMITS[i] for i in intervals]
            interval_limits = ", ".join(f"{i}->{max_lookback_label(INTERVAL_LIMITS[i])}" for i in intervals)
            full_interval_set = set(intervals) == set(INTERVAL_LIMITS.keys())
            interval_display = f"all ({len(intervals)})" if full_interval_set else interval
            max_cap_days = max(interval_max_days)
            default_range = max_lookback_label(max_cap_days)
            duration_raw = ask("range (10y / 60d / auto)", default_range)
            years_val = duration_to_years(duration_raw)
            if years_val is None or years_val <= 0:
                print("\n  [ERROR] Invalid range. Try 30d, 6mo, 2y, or auto.")
                pause()
                continue
            years_arg = f"{years_val:g}"
            caps_note = "per-interval Yahoo caps" if full_interval_set else interval_limits
            print(f"\n  → {', '.join(tickers)} | {interval_display} | {duration_raw} ({caps_note})")
            if ask_yes_no("Ready to download?", default=True):
                print("")
                run_script("get_stock_data.py", ["--ticker", *tickers, "--interval", *intervals, "--years", years_arg])
                pause()
            else:
                print("  Cancelled.")
                pause()
        else:
            print("\nInvalid choice, please try again")

