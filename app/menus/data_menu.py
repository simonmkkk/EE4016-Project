from ..actions import run_script
from ..constants import INTERVAL_LIMITS
from ..services import duration_to_years, max_lookback_label
from ..ui import ask, ask_yes_no, clear_screen, pause


def _print_supported_intervals() -> None:
    print("  Supported intervals (yfinance lookback limits):")
    for iv, max_days in INTERVAL_LIMITS.items():
        print(f"    {iv:4}  max {max_lookback_label(max_days)}")
    print(f"  All codes: {', '.join(INTERVAL_LIMITS.keys())}")
    print("  Type 'all' to use every interval above.")
    print("")


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
            print(f"  Selected tickers: {', '.join(tickers)}")
            print("")
            _print_supported_intervals()
            interval_raw = ask("interval(s) (e.g. 1d,1h,5m or all)", "1d,1h").lower()
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
            interval_display = (
                f"all ({len(intervals)} intervals, caps in table above)"
                if full_interval_set
                else interval
            )
            print(f"  Selected interval(s): {interval_display}")
            print("")
            range_hint = (
                "auto = clip each interval to its Yahoo max (see table above)"
                if full_interval_set
                else f"per-interval caps: {interval_limits}"
            )
            max_cap_days = max(interval_max_days)
            default_range = max_lookback_label(max_cap_days)
            print(
                "  One range sets requested history (in years internally); get_stock_data.py "
                "clips each interval to its own limit above. Default = as long as the longest "
                f"interval allows ({default_range}), not the shortest."
            )
            print("")
            duration_raw = ask(
                f"historical range (e.g. 10y, 60d, 6mo) or {range_hint}",
                default_range,
            )
            years_val = duration_to_years(duration_raw)
            if years_val is None or years_val <= 0:
                print("\n  [ERROR] Invalid duration format. Use 30d / 6mo / 2y.")
                pause()
                continue
            print(f"  Selected historical range: {duration_raw}")
            years_arg = f"{years_val:g}"
            plan_detail = "; ".join(
                (
                    f"{ticker} -> all intervals (per-interval caps)"
                    if full_interval_set
                    else f"{ticker} -> {interval} ({interval_limits})"
                )
                for ticker in tickers
            )
            print("\n" + "-" * 70)
            print("  Download Summary")
            print("-" * 70)
            print(f"  Tickers        : {', '.join(tickers)}")
            print(f"  Intervals      : {interval_display}")
            print(f"  Download Plan  : {plan_detail}")
            print(f"  Historical     : {duration_raw}")
            print("-" * 70)
            if ask_yes_no("Ready to download?", default=True):
                print("")
                run_script("get_stock_data.py", ["--ticker", *tickers, "--interval", *intervals, "--years", years_arg])
                pause()
            else:
                print("  Cancelled.")
                pause()
        else:
            print("\nInvalid choice, please try again")

