from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


def ensure_parent(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str | Path, payload: Any) -> Path:
    out = ensure_parent(path)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pickle(path: str | Path, payload: Any) -> Path:
    out = ensure_parent(path)
    with open(out, "wb") as f:
        pickle.dump(payload, f)
    return out
