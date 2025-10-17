import numpy as np

from src.statistics import _pack_stats_for_cache, _unpack_stats_from_cache


def test_statistics_cache_pack_round_trip():
    stats = {
        "style": "intraday",
        "timeframe": "5",
        "horizon_minutes": 60.0,
        "long": {"mfe": np.array([0.1, 0.2], dtype=np.float32), "quantiles": {"q50": 0.15}},
        "short": {"mfe": np.array([0.05], dtype=np.float32), "quantiles": {"q50": 0.05}},
        "atr": 1.2,
        "expected_move": 1.8,
    }

    packed = _pack_stats_for_cache(stats)
    assert "mfe" not in packed["long"]
    assert packed["long"]["mfe_count"] == 2

    restored = _unpack_stats_from_cache(packed)

    assert np.allclose(restored["long"]["mfe"], stats["long"]["mfe"])
    assert np.allclose(restored["short"]["mfe"], stats["short"]["mfe"])
