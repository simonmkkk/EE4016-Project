#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# Menu - Interactive menu for the stock ML system (FYP2-style)
# ============================================================================

from .menus import (
    run_data_menu,
    run_train_menu,
    run_backtest_menu,
    run_pipeline_menu,
    run_predict_menu,
    run_results_menu,
)
from .ui import clear_screen


# ============================================================================
# Interactive Menu - Dispatch Table
# ============================================================================
MENU_ACTIONS = {
    "1": run_data_menu,
    "2": run_train_menu,
    "3": run_backtest_menu,
    "4": run_predict_menu,
    "5": run_pipeline_menu,
    "6": run_results_menu,
}

def print_menu():
    """Print interactive menu.

    Shows all available options.
    """
    print("\n" + "=" * 70)
    print("Stock ML - Training & Backtesting System")
    print("=" * 70)
    print("\nPlease select an option:\n")

    print("[Data]")
    print("   1. Data menu")

    print("\n[Train]")
    print("   2. Train menu")

    print("\n[Backtest]")
    print("   3. Backtest menu")

    print("\n[Predict]")
    print("   4. Predict menu")

    print("\n[Pipeline]")
    print("   5. Pipeline menu")

    print("\n[Results]")
    print("   6. Results menu")

    print("\n   0. Exit")
    print("=" * 70)


def interactive_menu():
    """Interactive menu main loop.

    Runs until the user chooses to exit.
    """
    while True:
        clear_screen()
        print_menu()

        try:
            choice = input("\nEnter your choice: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if choice == "0":
            print("\nGoodbye!")
            break

        action = MENU_ACTIONS.get(choice)
        if action:
            action()
        else:
            print("\nInvalid choice, please try again")

