import pytest
import numpy as np
import pandas as pd

from pytaa.backtest.positions import (
    Positions, EqualWeights, vigilant_allocation, rolling_optimization, kipnis_allocation
)

TEST_ASSETS = ["A", "B", "C", "D"]
TEST_DATES = ["2010-01-31", "2022-02-28"]
TEST_DATA = pd.Series(dict(zip(TEST_ASSETS, [0.05, -0.1, 0.1, -0.5])))
TEST_RETURNS = pd.DataFrame(dict(zip(TEST_ASSETS, [0.2, -0.01, 0.1, -0.05])), index=[TEST_DATES[0]])
TEST_SIGNALS = pd.DataFrame(dict(zip(TEST_ASSETS, [0.2, 0.01, 0.1, 0.05])), index=[TEST_DATES[0]])


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


@pytest.mark.parametrize("ret, dts, func, lb, shr, sf, expected", [
    (TEST_RETURNS, [TEST_DATES[0]], np.array, 1, None, None, [0.25, 0.25, 0.25, 0.25]),
])
def test_rolling_optimization(ret, dts, func, lb, shr, sf, expected, mocker):
    """Test rolling optimization."""
    mock = mocker.patch("pytaa.tools.risk.Covariance._estimate", return_value=expected)
    ropt = rolling_optimization(ret, dts, func, lb, shr, sf)
    assert np.isclose(expected, ropt.values).all()


@pytest.mark.parametrize("ret, sig, dts, ca, ra, sa, k, expected", [
    (TEST_RETURNS, TEST_RETURNS, [TEST_DATES[0]], ["A", "B"], ["C"], ["D"], 1, [1]),
    (TEST_RETURNS, TEST_RETURNS, [TEST_DATES[0]], ["A", "D"], ["C"], ["B"], 1, [1]),
    (TEST_RETURNS, TEST_RETURNS, [TEST_DATES[0]], ["B", "D"], ["A"], ["C"], 1, [1]),
    # (TEST_RETURNS, TEST_SIGNALS, [TEST_DATES[0]], ["B", "D"], ["A"], ["C"], 1, [1]),
])
def test_kipnis_allocation(ret, sig, dts, ca, ra, sa, k, expected, mocker):
    """Test kipnis allocation."""
    mocker.patch("pytaa.tools.risk.weighted_covariance_matrix", return_value = None)
    mock_mvo = mocker.patch("pytaa.tools.risk.calculate_min_variance_portfolio")
    mock_mvo.return_value = np.array([0.25, 0.25, 0.25, 0.25]).reshape(-1, 1)
    ka = kipnis_allocation(ret, sig, dts, ca, ra, sa, k)
    assert np.isclose(expected, ka.values).all()
