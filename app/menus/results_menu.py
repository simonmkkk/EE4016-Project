from ..paths import RESULTS_DIR
from ..services.discovery import rel
from ..ui import clear_screen, pause


def _print_results_menu():
    print("\n" + "=" * 70)
    print("Stock ML - Results Menu")
    print("=" * 70)
    print("\nPlease select an option:\n")
    print("[Summary]")
    print(f"   1. Show protocol comparison summary ({rel(RESULTS_DIR)}/comparison_summary.csv)")
    print("\n   0. Back")
    print("=" * 70)


def _show_comparison_summary():
    summary_path = RESULTS_DIR / "comparison_summary.csv"
    if not summary_path.exists():
        print(f"\n  {rel(summary_path)} not found. Run protocol runner first.")
        return
    try:
        import pandas as pd

        df = pd.read_csv(summary_path)
    except Exception as e:
        print(f"\n  Failed to read {summary_path}: {e}")
        return

    for (widx, interval), g in df.groupby(["window_idx", "interval"]):
        top_ret = g.sort_values("avg_total_return", ascending=False).iloc[0]
        top_shr = g.sort_values("avg_sharpe", ascending=False).iloc[0]
        print(f"\n  --- window_idx={widx} interval={interval} ---")
        print(
            f"  best_total_return: {top_ret['strategy']} "
            f"(avg_total_return={top_ret['avg_total_return']:.4f}, avg_sharpe={top_ret['avg_sharpe']:.4f})"
        )
        print(
            f"  best_avg_sharpe  : {top_shr['strategy']} "
            f"(avg_sharpe={top_shr['avg_sharpe']:.4f}, avg_total_return={top_shr['avg_total_return']:.4f})"
        )


def run_results_menu():
    while True:
        clear_screen()
        _print_results_menu()
        try:
            choice = input("\nEnter your choice [1]: ").strip() or "1"
            print("")
        except (KeyboardInterrupt, EOFError):
            print("\nBack")
            break
        if choice == "0":
            break
        if choice == "1":
            _show_comparison_summary()
            pause()
        else:
            print("\nInvalid choice, please try again")

