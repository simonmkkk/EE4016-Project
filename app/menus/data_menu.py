from ..actions import run_script
from ..constants import INTERVAL_LIMITS
from ..services import duration_to_years, max_lookback_label
from ..ui import ask, ask_yes_no, clear_screen, pause


def _print_data_menu():
    print("\n" + "=" * 70)
    print("Stock ML - Data Menu")
    print("=" * 70)
    print("\nPlease select an option:\n")
    print("[Download]")
    print("   1. Download by ticker/interval/historical range (get_stock_data.py)")
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
            interval = ask("interval (e.g. 1d, 1h, 5m)", "1d").lower()
            if interval not in INTERVAL_LIMITS:
                print(f"\n  [ERROR] Unsupported interval: {interval}")
                print(f"  Supported: {', '.join(INTERVAL_LIMITS.keys())}")
                pause()
                continue
            print(f"  Selected interval: {interval}")
            print("")
            max_days = INTERVAL_LIMITS[interval]
            max_label = max_lookback_label(max_days)
            duration_raw = ask(f"historical range (e.g. 30d, 6mo, 2y), max {max_label}", max_label)
            years_val = duration_to_years(duration_raw)
            if years_val is None or years_val <= 0:
                print("\n  [ERROR] Invalid duration format. Use 30d / 6mo / 2y.")
                pause()
                continue
            print(f"  Selected historical range: {duration_raw}")
            years_arg = f"{years_val:g}"
            print("\n" + "-" * 70)
            print("  Download Summary")
            print("-" * 70)
            print(f"  Tickers        : {', '.join(tickers)}")
            print(f"  Interval       : {interval}")
            print(f"  Historical     : {duration_raw}")
            print(f"  Interval Max   : {max_label}")
            print("-" * 70)
            if ask_yes_no("Ready to download?", default=True):
                print("")
                run_script("get_stock_data.py", ["--ticker", *tickers, "--interval", interval, "--years", years_arg])
                pause()
            else:
                print("  Cancelled.")
                pause()
        else:
            print("\nInvalid choice, please try again")

