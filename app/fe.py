"""OHLCV technical indicators with windows clamped to series length (short CSV safe)."""

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    if n == 0:
        return df

    def clamp_w(default: int) -> int:
        return max(1, min(default, n))

    rsi_w = clamp_w(14)
    bb_w = clamp_w(20)
    atr_w = clamp_w(14)
    vma_w = clamp_w(20)
    slow = clamp_w(26)
    fast = max(1, min(12, slow))
    sign = max(1, min(9, slow))
    if fast >= slow:
        fast = max(1, slow - 1) if slow > 1 else 1

    df["rsi"] = RSIIndicator(df["close"], window=rsi_w).rsi()
    df["macd"] = MACD(
        df["close"], window_slow=slow, window_fast=fast, window_sign=sign
    ).macd_diff()
    df["bbw"] = BollingerBands(df["close"], window=bb_w).bollinger_wband()
    df["atr"] = AverageTrueRange(
        df["high"], df["low"], df["close"], window=atr_w
    ).average_true_range()
    df["vma20"] = SMAIndicator(df["volume"], vma_w).sma_indicator()
    df["v_ratio"] = df["volume"] / df["vma20"]
    df["body"] = (df["open"] - df["close"]).abs()
    df["range"] = df["high"] - df["low"]
    return df
