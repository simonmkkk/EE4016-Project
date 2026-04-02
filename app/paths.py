"""Project paths from environment (.env). All data/model/results roots are controlled here."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

STATE_PATH = PROJECT_ROOT / ".menu_state.json"


def _resolve_dir(key: str, default_relative: str) -> Path:
    raw = os.getenv(key, default_relative).strip().strip('"').strip("'")
    if not raw:
        raw = default_relative
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


# Folder roots (configure in .env — see .env_example)
SAVE_DIR = _resolve_dir("SAVE_DIR", "historical_data")
RESULTS_DIR = _resolve_dir("RESULTS_DIR", "backtest_results")
MODEL_DIR = _resolve_dir("MODEL_DIR", "model")
