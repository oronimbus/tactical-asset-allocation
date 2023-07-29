import pytest
import numpy as np
import pandas as pd

from pytaa.backtest.performance import Tearsheet

TEST_RETURNS = pd.DataFrame(
    data={"A": [0.01, 0.02, -0.03, 0.005, 0.01, 0.0001, 0.03, -0.03, -0.05, 0.02, -0.01, 0.01]},
    index=pd.date_range("2000-01-01","2000-01-12")
)
TEST_RETURNS.index.name = "Date"
TEST_TS = None


@pytest.mark.parametrize("returns, bm, expected", [
    (TEST_RETURNS, None, TEST_TS)
])
def test_tearsheet(returns, bm, expected):
    """Test tearsheet summary."""
    ts = Tearsheet(returns, bm).summary()
    print(ts)
    pass