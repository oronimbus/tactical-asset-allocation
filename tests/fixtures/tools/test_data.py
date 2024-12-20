import numpy as np
import pandas as pd
import pytest

from pytaa.tools.data import (
    find_ticker_currency,
    get_historical_dividends,
    get_historical_price_data,
    get_issue_currency_for_tickers,
    get_strategy_price_data,
    validate_tickers,
)


@pytest.mark.parametrize(
    "tickers, raise_issue, expected",
    [([], False, False), (["GOOG", None], False, False), (["MSFT", "FB"], False, True)],
)
def test_validate_tickers(tickers, raise_issue, expected):
    """Test validate tickers function."""
    assert validate_tickers(tickers, raise_issue) == expected


@pytest.mark.parametrize("tix, start, end, expected", [("AAPL", "2022-01-31", "2022-02-28", 0.22)])
def test_get_historical_dividends(tix, start, end, expected):
    """Test historical dividend request."""
    output = get_historical_dividends([tix], start, end)
    assert output.values[0] == expected


@pytest.mark.parametrize(
    "tix, fb, expected", [("AAPL", None, "USD"), ("DUMMY_TICKER", "EUR", "EUR")]
)
def test_find_ticker_currency(tix, fb, expected):
    """Test currency of ticker request."""
    output = find_ticker_currency(tix, fb)
    assert output == expected


@pytest.mark.parametrize(
    "tix, start, end, adj, expected", [
        ("AAPL", "2022-01-31", "2022-02-01", True, 171.963119),
        ("AAPL", "2022-01-31", "2022-02-01", False, 174.77999),
    ]
)
def test_get_historical_price_data(tix, start, end, adj, expected):
    """Test historical price data request."""
    output = get_historical_price_data([tix], start, end, adj)
    close = output["Close"][tix].values[0]
    assert np.isclose(close, expected)


@pytest.mark.parametrize(
    "tix, expected",
    [
        ("AAPL", "USD"),
    ],
)
def test_get_issue_currency_for_tickers(tix, expected):
    """Test issue currency request."""
    output = get_issue_currency_for_tickers([tix])
    assert output[0] == expected


@pytest.mark.parametrize(
    "pipe, start, end, expected",
    [
        ([None], "2022-01-31", "2022-02-01", pd.DataFrame({"Close": [100]})),
    ],
)
def test_get_strategy_price_data(pipe, start, end, expected, mocker):
    """Test strategy price request."""
    # need to think about how to mock attributes of pipeline w/o passing pipeline
    assert 1 == 1
