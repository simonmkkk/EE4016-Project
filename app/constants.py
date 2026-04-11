INTERVAL_LIMITS = {
    "1m": 7,
    "2m": 60,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "60m": 730,
    "90m": 60,
    "1h": 730,
    "1d": 3650,
    "5d": 3650,
    "1wk": 3650,
    "1mo": 3650,
    "3mo": 3650,
}

# Stable 0..n-1 ids for models (order matches INTERVAL_LIMITS insertion order).
INTERVAL_ID_ORDER: tuple[str, ...] = tuple(INTERVAL_LIMITS.keys())
INTERVAL_NAME_TO_ID: dict[str, int] = {name: i for i, name in enumerate(INTERVAL_ID_ORDER)}
INTERVAL_ID_UNKNOWN: int = len(INTERVAL_ID_ORDER)


def interval_id_from_csv_stem(stem: str) -> int:
    """
    Parse yfinance interval token from filename stem.

    Supports:
      - New: TICKER_interval_lookback (e.g. AAPL_1d_10y)
      - Legacy: TICKER_lookback_interval (e.g. AAPL_10y_1d)
    """
    parts = stem.split("_")
    for j in (1, 2):
        if len(parts) > j:
            tok = parts[j].strip().lower()
            if tok in INTERVAL_NAME_TO_ID:
                return INTERVAL_NAME_TO_ID[tok]
    return INTERVAL_ID_UNKNOWN

