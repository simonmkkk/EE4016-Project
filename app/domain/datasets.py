from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import interval_id_from_csv_stem
from ..fe import add_technical_indicators


def load_price_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    return add_granularity_flag(df.sort_values("date"))


def add_granularity_flag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["granularity"] = (
        (out["date"].dt.hour != 0)
        | (out["date"].dt.minute != 0)
        | (out["date"].dt.second != 0)
    ).astype(np.int8)
    return out


def build_training_frame(
    df: pd.DataFrame,
    *,
    label_threshold: float = 0.0,
    label_threshold_quantile: float | None = None,
    series_id: str | None = None,
    interval_id: int | None = None,
) -> pd.DataFrame:
    out = add_technical_indicators(df)
    out["log_ret"] = np.log(out["close"]).diff().shift(-1)
    if label_threshold_quantile is not None:
        actual_thr = float(
            np.nanquantile(out["log_ret"].dropna().values, label_threshold_quantile)
        )
    else:
        actual_thr = float(label_threshold)
    out["direction"] = (out["log_ret"] > actual_thr).astype(np.float32)
    out["_label_thr_used"] = np.float32(actual_thr)
    if series_id is not None:
        out["series_id"] = series_id
    if interval_id is not None:
        out["interval_id"] = np.float32(interval_id)
    return out.dropna().reset_index(drop=True)


def build_prediction_frame(
    df: pd.DataFrame,
    *,
    keep_last_bar: bool,
) -> pd.DataFrame:
    out = add_technical_indicators(df)
    out["log_ret"] = np.log(out["close"]).diff().shift(-1)
    out["direction"] = np.where(
        out["log_ret"].notna(),
        (out["log_ret"] > 0).astype(np.float32),
        np.nan,
    )
    if keep_last_bar:
        drop_subset = [c for c in out.columns if c not in ("log_ret", "direction")]
        return out.dropna(subset=drop_subset).reset_index(drop=True)
    return out.dropna().reset_index(drop=True)


def build_backtest_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = add_technical_indicators(df)
    out["fwd_ret"] = out["close"].pct_change().shift(-1)
    return out.dropna().reset_index(drop=True)


def infer_symbol_from_path(path: str | Path) -> str:
    return Path(path).stem.split("_")[0].upper()


def infer_interval_id_from_path(path: str | Path) -> int:
    return interval_id_from_csv_stem(Path(path).stem)


def read_training_frame(
    path: str | Path,
    *,
    label_threshold: float = 0.0,
    label_threshold_quantile: float | None = None,
) -> pd.DataFrame:
    csv_path = Path(path)
    return build_training_frame(
        load_price_csv(csv_path),
        label_threshold=label_threshold,
        label_threshold_quantile=label_threshold_quantile,
        series_id=csv_path.stem,
        interval_id=infer_interval_id_from_path(csv_path),
    )


def read_prediction_frame(
    path: str | Path,
    *,
    keep_last_bar: bool = True,
) -> pd.DataFrame:
    csv_path = Path(path)
    df = build_prediction_frame(load_price_csv(csv_path), keep_last_bar=keep_last_bar)
    df["interval_id"] = np.float32(infer_interval_id_from_path(csv_path))
    return df


def read_backtest_frame(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    df = build_backtest_frame(load_price_csv(csv_path))
    df["interval_id"] = np.float32(infer_interval_id_from_path(csv_path))
    return df
