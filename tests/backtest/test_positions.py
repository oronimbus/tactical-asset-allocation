import pytest
import numpy as np
import pandas as pd

from pytaa.backtest.positions import Positions, EqualWeights, vigilant_allocation

TEST_ASSETS = ["A", "B", "C", "D"]
TEST_DATES = ["2010-01-31", "2022-02-28"]
TEST_DATA = pd.Series(dict(zip(TEST_ASSETS, [0.05, -0.1, 0.1, -0.5])))


@pytest.mark.parametrize("assets, dates, expected", [
    (TEST_ASSETS, TEST_DATES, 8)   
])
def test_positions(assets, dates, expected):
    """Test base class for Positions."""
    pos = Positions(assets, dates)
    assert pos.n_assets == len(assets)
    assert pos.n_obs == len(dates)
    assert len(pos.weights) == expected


@pytest.mark.parametrize("assets, dates, expected", [
    (TEST_ASSETS, TEST_DATES, 1 / 4)   
])
def test_equal_weights(assets, dates, expected):
    """Test Equal Weights class.."""
    ew = EqualWeights(assets, dates)
    assert np.isclose(ew.weights, expected).all()


@pytest.mark.parametrize("data, ra, sa, k, step, expected", [
    (TEST_DATA, ["A", "B"], ["C", "D"], 2, 0.5, np.array([0, 0, 1, 0])), 
    (TEST_DATA.abs(), ["A", "B"], ["C", "D"], 2, 0.5, np.array([0.5, 0.5, 0, 0])),
    (TEST_DATA.abs(), ["A", "B", "C"], ["D"], 3, 1/3, np.array([1/3, 1/3, 1/3, 0])),
    # this case is a bit weird since we have 2 negatives but only one safe asset
    (TEST_DATA, ["A", "B", "C"], ["D"], 3, 1/3, np.array([1/9, 1/9, 1/9, 2/3]))
])
def test_vigilant_allocation(data, ra, sa, k, step, expected):
    """Test Vigilant allocation."""
    va = vigilant_allocation(data, ra, sa, k, step).values
    assert va.sum() == 1
    assert np.isclose(va, expected).all()
