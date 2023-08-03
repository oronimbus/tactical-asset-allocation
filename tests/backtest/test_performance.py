import pytest
import numpy as np
import pandas as pd

from pytaa.backtest.performance import Tearsheet

TEST_RETURNS = pd.DataFrame(
    data={"A": [0.01, 0.02, -0.03, 0.005, 0.01, 0.0001, 0.03, -0.03, -0.05, 0.02, -0.01, 0.01]},
    index=pd.date_range("2000-01-01", "2000-01-12"),
)
TEST_RETURNS.index.name = "Date"
TEST_TS = {
    "#obs": 12,
    "#years": 0.030116358658453114,
    "Total Return": -0.017995569871820605,
    "Annual Return": -0.45281952334930353,
    "Volatility": 0.08359975532804562,
    "Max Drawdown": -0.07850000000000013,
    "Skewness": -0.8536696648152846,
    "Kurtosis": 2.8851688725498037,
    "Sharpe Ratio": -0.17823018669770466,
    "Standard Error": 1.0006236000757347,
    "Sortino": -0.26339727599198887,
    "Calmar": 5.768401571328698,
    "HWM": 1.044867852278594,
    "Auto Correlation": -0.19978232211975622,
}


@pytest.mark.parametrize("returns, bm, expected", [(TEST_RETURNS, None, TEST_TS)])
def test_tearsheet(returns, bm, expected):
    """Test tearsheet summary."""
    ts = Tearsheet(returns, bm).summary()
    for key in expected.keys():
        assert np.isclose(ts.loc[key].values[0], expected[key])
