import pytest
import numpy as np
import pandas as pd

from pytaa.tools.utils import (
    autocorrelation, calculate_rolling_volatility, calculate_risk_parity, risk_contribution
)


n_sample = 10
index = pd.bdate_range("2011-01-01", "2011-01-31")
SAMPLE_DATA = pd.DataFrame(np.arange(0, n_sample), columns=["A"], index=index[:n_sample])
SAMPLE_DATA.index.name = "Date"

SAMPLE_CORR = np.array([
    [1, 0.8, 0, 0],
    [0.8, 1, 0, 0],
    [0, 0, 1, -0.5],
    [0, 0, -0.5, 1]
])

SAMPLE_VOL = np.array([0.1, 0.2, 0.3, 0.4])
SAMPLE_VCV = np.diag(SAMPLE_VOL) @ SAMPLE_CORR @ np.diag(SAMPLE_VOL)


@pytest.mark.parametrize("returns, order, expected", [(SAMPLE_DATA, 1, 1)])
def test_autocorrelation(returns, order, expected):
    """Test autocorrelation function."""
    assert autocorrelation(returns, order)[0] == expected


@pytest.mark.parametrize("returns, lookback, factor, estimator, decay, expected", [
    (SAMPLE_DATA, 5, 1, "hist", None, 0.04958444090685402),
    (SAMPLE_DATA, 5, 1, "ewm", 0.99, 0.2906118375739968)
])
def test_calculate_rolling_volatility(returns, lookback, factor, estimator, decay, expected):
    """Test rolling volatility estimator."""
    series = returns.pct_change().replace(np.inf, np.nan).dropna()
    results = calculate_rolling_volatility(series, lookback, factor, estimator, decay)
    assert np.isclose(results.values[-1][0], expected)


@pytest.mark.parametrize("cov, expected", [
    (SAMPLE_VCV, [0.3917958239342683, 0.19589800250631811, 0.2477934771187355, 0.18584501063779774])
])
def test_risk_parity(cov, expected):
    """Test risk parity weights."""
    w_opt = calculate_risk_parity(cov)
    assert np.allclose(list(w_opt.flatten()), expected)


@pytest.mark.parametrize("cov", [(SAMPLE_VCV)])
def test_risk_contribution(cov):
    """Test risk contribution using risk parity."""
    w_opt = calculate_risk_parity(cov)
    risk_contrib = risk_contribution(w_opt, cov)
    for i in range(len(risk_contrib) - 1):
        assert np.isclose(risk_contrib[i], risk_contrib[i+1])



