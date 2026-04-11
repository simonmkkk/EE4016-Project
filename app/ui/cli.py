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


def choose_from_list(title: str, items: list[str], *, default_index: int | None = None, header: str | None = None) -> str | None:
    if not items:
        print(f"\n[WARN] No items found for: {title}")
        return None
    if len(items) == 1:
        return items[0]

    def _render_menu() -> None:
        clear_screen()
        if header:
            print(header)
        print(f"\n{title}")
        for i, it in enumerate(items, start=1):
            print(f"  {i}. {it}")
        print("  0. Cancel")

    _render_menu()
    d = str(default_index) if isinstance(default_index, int) else ""
    while True:
        raw = ask("Enter choice", d).strip()
        if raw == "0":
            return None
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(items):
                return items[idx - 1]
        print("  Invalid choice, please try again.")
        _render_menu()


def choose_multiple_from_list(title: str, items: list[str], *, header: str | None = None) -> list[str] | None:
    if not items:
        print(f"\n[WARN] No items found for: {title}")
        return None
    if len(items) == 1:
        return [items[0]]

    def _render_menu(error_message: str | None = None) -> None:
        clear_screen()
        if header:
            print(header)
        if error_message:
            print(f"  {error_message}\n")
        print(f"\n{title}")
        for i, it in enumerate(items, start=1):
            print(f"  {i}. {it}")
        print("  0. Cancel")
        print("  a. All")
        print("  Enter choices separated by comma or range, e.g. 1,3-5")

    _render_menu()
    while True:
        raw = ask("Enter choice(s)").strip().lower()
        if raw == "0":
            return None
        if raw == "a" or raw == "all":
            return items[0:]
        choices = []
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if not parts:
            _render_menu("Invalid choice(s), please try again.")
            continue
        valid = True
        for part in parts:
            if "-" in part:
                bounds = part.split("-")
                if len(bounds) != 2 or not all(x.isdigit() for x in bounds):
                    valid = False
                    break
                start, end = int(bounds[0]), int(bounds[1])
                if start < 1 or end > len(items) or start > end:
                    valid = False
                    break
                choices.extend(range(start, end + 1))
            elif part.isdigit():
                idx = int(part)
                if idx < 1 or idx > len(items):
                    valid = False
                    break
                choices.append(idx)
            else:
                valid = False
                break
        if not valid:
            _render_menu("Invalid choice(s), please try again.")
            continue
        unique_indices = sorted(dict.fromkeys(choices))
        return [items[i - 1] for i in unique_indices]

