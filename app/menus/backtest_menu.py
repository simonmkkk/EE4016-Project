from ..actions import run_backtest_pick_csv_and_model
from ..ui import clear_screen, pause


def _print_backtest_menu():
    print("\n" + "=" * 70)
    print("Stock ML - Backtest Menu")
    print("=" * 70)
    print("\nPlease select an option:\n")
    print("[Backtest]")
    print("   1. Backtest (pick one or more CSVs + model) (backtest_stock.py)")
    print("\n   0. Back")
    print("=" * 70)


def run_backtest_menu():
    while True:
        clear_screen()
        _print_backtest_menu()
        try:
            choice = input("\nEnter your choice [1]: ").strip() or "1"
            print("")
        except (KeyboardInterrupt, EOFError):
            print("\nBack")
            break
        if choice == "0":
            break
        if choice == "1":
            run_backtest_pick_csv_and_model()
            pause()
        else:
            print("\nInvalid choice, please try again")

