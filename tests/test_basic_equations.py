import pytest
from numpy import exp
from scipy import stats
from compute.basic_equations import get_Fcoll, repairtime_function

# FILE: compute/test_basic_equations.py


def test_get_Fcoll():
    assert get_Fcoll(10, 0.1) == 1.0
    assert get_Fcoll(0, 0.1) == 0.0
    assert get_Fcoll(10, 0) == 0.0

def test_repairtime_function():
    data = {
        "active_window": 0,
        "std": 0.5,
        "loc": 0,
        "scale": 1
    }
    x = 1
    expected_result = stats.lognorm(data["std"], data["loc"], data["scale"]).cdf(x)
    assert repairtime_function(data, x) == expected_result

    data["active_window"] = 1
    data["func"] = "x * 2"
    expected_result = eval(data["func"])
    assert repairtime_function(data, x) == expected_result