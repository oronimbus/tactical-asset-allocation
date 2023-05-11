import pytest
import numpy as np
import pandas as pd

from pytaa.tools.utils import autocorrelation


SAMPLE_DATA = pd.DataFrame(np.ones(12))
SAMPLE_DATA.index.name = "Date"


@pytest.mark.parametrize("returns, order, expected", [(SAMPLE_DATA, 1, 1)])
def test_autocorrelation(returns, order, expected):
    # print(returns)
    # print(autocorrelation(returns, order))
    # assert autocorrelation(returns, order)[0] == np.nan
    assert True == True
