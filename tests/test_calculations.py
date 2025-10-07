import numpy as np
import pandas as pd

from src.calculations import atr, bollinger_bands, ema, keltner_channels, vwap


def test_ema_simple_series() -> None:
    series = pd.Series([1, 2, 3, 4, 5], dtype=float)
    result = ema(series, period=3)
    expected = np.array([1.0, 1.5, 2.25, 3.125, 4.0625])
    assert np.allclose(result.values, expected, atol=1e-6)


def test_atr_constant_true_range() -> None:
    high = pd.Series([10, 11, 12, 13, 14], dtype=float)
    low = pd.Series([8, 9, 10, 11, 12], dtype=float)
    close = pd.Series([9, 10, 11, 12, 13], dtype=float)
    result = atr(high, low, close, period=3)
    # With a constant true range of 2, ATR should also remain at 2
    assert np.allclose(result.dropna().values, 2.0, atol=1e-6)


def test_vwap_weighted_average() -> None:
    price = pd.Series([10, 11, 12], dtype=float)
    volume = pd.Series([1, 2, 3], dtype=float)
    result = vwap(price, volume)
    expected = np.array([10.0, 32 / 3, 68 / 6])
    assert np.allclose(result.values, expected, atol=1e-6)


def test_bollinger_bands_known_values() -> None:
    prices = pd.Series([1, 2, 3, 4, 5], dtype=float)
    upper, lower = bollinger_bands(prices, period=3, width=2.0)
    expected_upper = np.array([np.nan, np.nan, 3.632993, 4.632993, 5.632993])
    expected_lower = np.array([np.nan, np.nan, 0.367007, 1.367007, 2.367007])
    assert np.allclose(upper.values, expected_upper, equal_nan=True, atol=1e-6)
    assert np.allclose(lower.values, expected_lower, equal_nan=True, atol=1e-6)


def test_keltner_channels_constant_series() -> None:
    prices = pd.Series([10, 10, 10, 10], dtype=float)
    high = pd.Series([10.5, 10.5, 10.5, 10.5], dtype=float)
    low = pd.Series([9.5, 9.5, 9.5, 9.5], dtype=float)
    upper, lower = keltner_channels(prices, high, low, period=2, atr_factor=1.5)
    # Constant prices produce a constant EMA, while the ATR should stay at 1.0
    assert np.allclose(upper.dropna().values, 11.5, atol=1e-6)
    assert np.allclose(lower.dropna().values, 8.5, atol=1e-6)
