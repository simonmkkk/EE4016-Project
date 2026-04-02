import json

from .paths import STATE_PATH


def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state: dict) -> None:
    # Respect user's preference: do not auto-create .menu_state.json.
    # Persist only when the file already exists.
    if not STATE_PATH.exists():
        return
    try:
        STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

