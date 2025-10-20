import numpy as np

from src.agent_server import _nativeify


def test_nativeify_converts_numpy_scalars():
    payload = {
        "int_val": np.int64(7),
        "float_list": [np.float32(1.25), {"nested": np.int32(3)}],
        "tuple_payload": (np.float64(2.5), np.int16(4)),
    }

    result = _nativeify(payload)

    assert isinstance(result["int_val"], int)
    assert isinstance(result["float_list"][0], float)
    assert isinstance(result["float_list"][1]["nested"], int)
    assert isinstance(result["tuple_payload"][0], float)
    assert isinstance(result["tuple_payload"][1], int)
