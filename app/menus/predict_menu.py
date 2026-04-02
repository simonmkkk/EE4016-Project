from ..actions import run_predict_pick_csv_and_model
from ..ui import clear_screen, pause


def _print_predict_menu():
    print("\n" + "=" * 70)
    print("Stock ML - Predict Menu")
    print("=" * 70)
    print("\nPlease select an option:\n")
    print("[Predict]")
    print("   1. Predict + explanation (pick CSV + model) (predict_stock.py)")
    print("\n   0. Back")
    print("=" * 70)


def run_predict_menu():
    while True:
        clear_screen()
        _print_predict_menu()
        try:
            choice = input("\nEnter your choice [1]: ").strip() or "1"
            print("")
        except (KeyboardInterrupt, EOFError):
            print("\nBack")
            break
        if choice == "0":
            break
        if choice == "1":
            run_predict_pick_csv_and_model()
            pause()
        else:
            print("\nInvalid choice, please try again")

