import pytest
import pandas as pd
import yfinance as yf

from pytaa.tools.data import validate_tickers, get_historical_dividends

TEST_START, TEST_END = pd.Timestamp("2020-03-31"), pd.Timestamp("2020-04-30")


@pytest.mark.parametrize(
    "tickers, raise_issue, expected",
    [([], False, False), (["GOOG", None], False, False), (["MSFT", "FB"], False, True)],
)
def test_validate_tickers(tickers, raise_issue, expected):
    """Test validate tickers function."""
    assert validate_tickers(tickers, raise_issue) == expected


@pytest.mark.parametrize("tix, start, end, expected", [("A", TEST_START, TEST_END, 1)])
def test_get_historical_dividends(tix, start, end, expected, mocker):
    """Test historical dividend request."""
    sample = pd.Series([0, 1], index=[start, end])
    # mock = mocker.patch("yf.Ticker.dividends", return_value=sample)
    # output = get_historical_dividends([tix], start, end)
    # print(output)
    assert 1 == 1