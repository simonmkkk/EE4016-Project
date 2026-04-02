import os


def ask(prompt: str, default: str | None = None) -> str:
    if default is None:
        v = input(f"  {prompt}: ").strip()
    else:
        v = input(f"  {prompt} [default={default}]: ").strip()
    return v.strip('"').strip("'") or (default or "")


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    raw = input(f"  {prompt} [{d}]: ").strip().lower()
    if not raw:
        return default
    return raw.startswith("y")


def pause() -> None:
    input("\n  Press Enter to continue...")


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def choose_from_list(title: str, items: list[str], *, default_index: int | None = None) -> str | None:
    if not items:
        print(f"\n[WARN] No items found for: {title}")
        return None
    if len(items) == 1:
        return items[0]

    print(f"\n{title}")
    for i, it in enumerate(items, start=1):
        print(f"  {i}. {it}")
    print("  0. Cancel")

    d = str(default_index) if isinstance(default_index, int) else ""
    while True:
        raw = ask("Enter choice", d)
        if raw == "0":
            return None
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(items):
                return items[idx - 1]
        print("  Invalid choice, please try again.")

