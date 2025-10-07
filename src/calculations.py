"""Indicator calculations used by the trading assistant.

This module implements several commonly used technical indicators such as
Average True Range (ATR), Exponential Moving Average (EMA), Volume-Weighted
Average Price (VWAP), and the Average Directional Index (ADX).  These
functions operate on pandas Series or DataFrame objects.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def ema(series: pd.Series, period: int) -> pd.Series:
    """Compute the Exponential Moving Average (EMA) of a series.

    Args:
        series: A pandas Series of values.
        period: The lookback period for the EMA.

    Returns:
        A pandas Series containing the EMA values.
    """
    return series.ewm(span=period, adjust=False).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Average True Range (ATR).

    ATR is calculated as the exponential moving average of the True Range.

    Args:
        high: Series of high prices.
        low: Series of low prices.
        close: Series of closing prices.
        period: Lookback period. Default is 14.

    Returns:
        A pandas Series of ATR values.
    """
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return ema(true_range, period)


def vwap(price: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute the Volume‑Weighted Average Price (VWAP).

    VWAP is calculated as the cumulative sum of price * volume divided by the
    cumulative volume.  The price parameter should typically be the typical
    price (high + low + close) / 3 or simply the closing price.

    Args:
        price: Series of prices (e.g. typical price or close).
        volume: Series of trade volumes.

    Returns:
        A pandas Series of VWAP values.
    """
    cum_volume = volume.cumsum()
    cum_vp = (price * volume).cumsum()
    return cum_vp / cum_volume


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Average Directional Index (ADX).

    ADX quantifies trend strength by combining the positive and negative
    directional indicators (+DI and −DI).  A higher ADX indicates a stronger
    trend, whereas lower values suggest consolidation or range‑bound price
    action.

    Args:
        high: Series of high prices.
        low: Series of low prices.
        close: Series of closing prices.
        period: Lookback period for DI and ADX calculation (default 14).

    Returns:
        A pandas Series representing the ADX.
    """
    delta_high = high.diff()
    delta_low = low.diff()
    pos_dm = delta_high.where((delta_high > delta_low) & (delta_high > 0), 0.0)
    neg_dm = (-delta_low).where((delta_low > delta_high) & (delta_low > 0), 0.0)
    tr = atr(high, low, close, period)
    pos_di = 100 * ema(pos_dm, period) / tr
    neg_di = 100 * ema(neg_dm, period) / tr
    dx = (pos_di - neg_di).abs() / (pos_di + neg_di) * 100
    adx_series = ema(dx, period)
    return adx_series


def bollinger_bands(series: pd.Series, period: int = 20, width: float = 2.0) -> tuple[pd.Series, pd.Series]:
    """Compute Bollinger Bands (upper and lower) around a moving average.

    Args:
        series: Series of closing prices.
        period: Lookback period for the moving average (default 20).
        width: Number of standard deviations for the bands (default 2.0).

    Returns:
        A tuple `(upper_band, lower_band)` as pandas Series.
    """
    ma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std(ddof=0)
    upper = ma + width * std
    lower = ma - width * std
    return upper, lower


def keltner_channels(series: pd.Series, high: pd.Series, low: pd.Series, period: int = 20, atr_factor: float = 1.5) -> tuple[pd.Series, pd.Series]:
    """Compute Keltner Channels (upper and lower) around an EMA and ATR.

    Args:
        series: Series of closing prices (for the middle EMA).
        high: Series of high prices.
        low: Series of low prices.
        period: Lookback period for the EMA and ATR (default 20).
        atr_factor: Multiplier for the ATR to set channel width (default 1.5).

    Returns:
        A tuple `(upper_channel, lower_channel)` as pandas Series.
    """
    ema_mid = ema(series, period)
    atr_val = atr(high, low, series, period)
    upper = ema_mid + atr_factor * atr_val
    lower = ema_mid - atr_factor * atr_val
    return upper, lower