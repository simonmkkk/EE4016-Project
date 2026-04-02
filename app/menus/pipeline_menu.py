from ..actions import run_protocol_runner
from ..ui import clear_screen, pause


def _print_pipeline_menu():
    print("\n" + "=" * 70)
    print("Stock ML - Pipeline Menu")
    print("=" * 70)
    print("\nPlease select an option:\n")
    print("[Pipeline]")
    print("   1. Run protocol runner (run_protocol.py)")
    print("\n   0. Back")
    print("=" * 70)


def run_pipeline_menu():
    while True:
        clear_screen()
        _print_pipeline_menu()
        try:
            choice = input("\nEnter your choice [1]: ").strip() or "1"
            print("")
        except (KeyboardInterrupt, EOFError):
            print("\nBack")
            break
        if choice == "0":
            break
        if choice == "1":
            run_protocol_runner()
            pause()
        else:
            print("\nInvalid choice, please try again")

