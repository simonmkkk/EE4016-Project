from pathlib import Path

from ..constants import interval_id_from_csv_stem
from ..paths import PROJECT_ROOT, SAVE_DIR, MODEL_DIR
from ..state import load_state, save_state
from ..ui import choose_from_list, choose_multiple_from_list


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except Exception:
        return str(p)


def list_historical_csvs() -> list[Path]:
    rec = SAVE_DIR
    if not rec.exists():
        return []

    def sort_key(p: Path) -> tuple[int, str]:
        # interval_id matches INTERVAL_ID_ORDER (shortest→longest); unknown last
        return (interval_id_from_csv_stem(p.stem), p.stem.lower())

    return sorted(rec.glob("*.csv"), key=sort_key)


def list_model_pts() -> list[Path]:
    mdl = MODEL_DIR
    return sorted(mdl.glob("*/*.pt"), key=lambda p: str(p).lower()) if mdl.exists() else []


def list_protocol_jsons() -> list[Path]:
    js = sorted(PROJECT_ROOT.glob("*.json"), key=lambda p: p.name.lower())

    def score(p: Path) -> tuple[int, str]:
        n = p.name.lower()
        return (0 if "protocol" in n else 1, n)

    return sorted(js, key=score)


def infer_ticker_from_csv(csv_path: Path) -> str | None:
    parts = csv_path.stem.split("_")
    tic = parts[0].strip().upper() if parts else ""
    return tic or None


def max_lookback_label(max_days: int) -> str:
    if max_days >= 365:
        return f"{round(max_days / 365)}y"
    return f"{max_days}d"


def duration_to_years(text: str) -> float | None:
    raw = text.strip().lower()
    if not raw:
        return None
    # Large lookback so get_stock_data.py can clip each interval to its own yfinance cap.
    if raw in ("auto", "max", "maximum", "full"):
        return 100.0
    if raw.endswith("mo") and raw[:-2].strip().replace(".", "", 1).isdigit():
        return float(raw[:-2].strip()) / 12
    if raw.endswith("d") and raw[:-1].strip().replace(".", "", 1).isdigit():
        return float(raw[:-1].strip()) / 365
    if raw.endswith("y") and raw[:-1].strip().replace(".", "", 1).isdigit():
        return float(raw[:-1].strip())
    if raw.replace(".", "", 1).isdigit():
        return float(raw)
    return None


def pick_csv(*, state_key: str = "last_csv") -> Path | None:
    state = load_state()
    csvs = list_historical_csvs()
    items = [rel(p) for p in csvs]
    last = state.get(state_key)
    default_idx = (items.index(last) + 1) if (last in items) else None
    chosen = choose_from_list(f"  Select CSV ({rel(SAVE_DIR)}/*.csv)", items, default_index=default_idx)
    if not chosen:
        return None
    state[state_key] = chosen
    save_state(state)
    return PROJECT_ROOT / chosen


def pick_csvs(
    *,
    state_key: str = "last_csvs",
    header: str | None = None,
    list_title: str | None = None,
) -> list[Path] | None:
    state = load_state()
    csvs = list_historical_csvs()
    if not csvs:
        return None

    # If multiple tickers exist, let the user filter by ticker first.
    ticker_to_csvs: dict[str, list[Path]] = {}
    for p in csvs:
        tic = infer_ticker_from_csv(p) or "UNKNOWN"
        ticker_to_csvs.setdefault(tic, []).append(p)

    selected_csvs = csvs
    if len(ticker_to_csvs) > 1:
        sorted_tickers = sorted(ticker_to_csvs.keys())
        filter_items = [f"ALL ({len(csvs)} files)"] + [
            f"{tic} ({len(ticker_to_csvs[tic])} files)" for tic in sorted_tickers
        ]
        chosen_filter = choose_from_list(
            "  Select ticker filter",
            filter_items,
            header=header,
        )
        if not chosen_filter:
            return None
        if not chosen_filter.startswith("ALL "):
            chosen_ticker = chosen_filter.split(" ", 1)[0]
            selected_csvs = ticker_to_csvs.get(chosen_ticker, [])
            if not selected_csvs:
                return None

    items = [rel(p) for p in selected_csvs]
    title = list_title or f"  Select one or more CSVs ({rel(SAVE_DIR)}/*.csv)"
    chosen_items = choose_multiple_from_list(
        title,
        items,
        header=header,
    )
    if not chosen_items:
        return None
    state[state_key] = chosen_items[0]
    save_state(state)
    return [PROJECT_ROOT / item for item in chosen_items]


def pick_model(*, prefer_ticker: str | None = None, state_key: str = "last_model") -> Path | None:
    state = load_state()
    pts = list_model_pts()
    if prefer_ticker:
        pts_pref = [p for p in pts if p.parent.name.upper() == prefer_ticker.upper()]
        pts = pts_pref or pts
    items = [rel(p) for p in pts]
    last = state.get(state_key)
    default_idx = (items.index(last) + 1) if (last in items) else None
    chosen = choose_from_list(f"  Select model ({rel(MODEL_DIR)}/*/*.pt)", items, default_index=default_idx)
    if not chosen:
        return None
    state[state_key] = chosen
    save_state(state)
    return PROJECT_ROOT / chosen


def pick_protocol(*, state_key: str = "last_protocol") -> Path | None:
    state = load_state()
    js = list_protocol_jsons()
    items = [rel(p) for p in js]
    last = state.get(state_key, "experiment_protocol.json")
    default_idx = (items.index(last) + 1) if (last in items) else None
    chosen = choose_from_list("  Select protocol (*.json)", items, default_index=default_idx)
    if not chosen:
        return None
    state[state_key] = chosen
    save_state(state)
    return PROJECT_ROOT / chosen

