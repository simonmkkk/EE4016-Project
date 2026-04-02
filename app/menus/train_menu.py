from ..actions import run_train_from_historical_csv
from ..ui import clear_screen, pause


def _print_train_menu():
    print("\n" + "=" * 70)
    print("Stock ML - Train Menu")
    print("=" * 70)
    print("\nPlease select an option:\n")
    print("[Train]")
    print("   1. Train from historical data CSV (pick CSV -> infer ticker) (train_stock.py)")
    print("\n   0. Back")
    print("=" * 70)


def run_train_menu():
    while True:
        clear_screen()
        _print_train_menu()
        try:
            choice = input("\nEnter your choice [1]: ").strip() or "1"
            print("")
        except (KeyboardInterrupt, EOFError):
            print("\nBack")
            break
        if choice == "0":
            break
        if choice == "1":
            run_train_from_historical_csv()
            pause()
            break
        else:
            print("\nInvalid choice, please try again")

