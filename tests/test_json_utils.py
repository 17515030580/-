import math
import numpy as np

from app.utils.json_utils import json_safe


def test_json_safe_non_finite():
    result = json_safe({"a": np.float32(1.5), "b": math.nan})
    assert result == {"a": 1.5, "b": None}
