import pytest
import numpy as np
import pandas as pd

from pytaa.tools.utils import autocorrelation, calculate_rolling_volatility

n_sample = 10
index = pd.bdate_range("2011-01-01", "2011-01-31")
SAMPLE_DATA = pd.DataFrame(np.arange(0, n_sample), columns=["A"], index=index[:n_sample])
SAMPLE_DATA.index.name = "Date"


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
