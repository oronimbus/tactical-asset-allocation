import pandas as pd
import pytest

from pytaa.strategy.signals import Signal

TEST_PRICES = pd.DataFrame(
    {"p": [100, 102, 101, 104, 103, 101, 99, 98, 97, 98, 95, 94, 96]},
    index=pd.bdate_range("2010-01-01", "2011-02-01", freq="BME"),
)


@pytest.mark.parametrize(
    "prices, start, end, expected",
    [
        (TEST_PRICES, 12, 1, 94 / 100 - 1),
        (TEST_PRICES, 7, 2, 95 / 101 - 1),
    ],
)
def test_signal_momentum(prices, start, end, expected):
    """Test momentum of Signal class."""
    obj = Signal(prices).classic_momentum(start, end)
    assert obj.values[-1, 0] == expected


@pytest.mark.parametrize(
    "prices, expected",
    [
        (TEST_PRICES, 12 * 96 / 94 + 4 * 96 / 98 + 2 * 96 / 99 + 96 / 100 - 19),
    ],
)
def test_signal_momentum_score(prices, expected):
    """Test simple momentum of Signal class."""
    obj = Signal(prices).momentum_score()
    assert obj.values[-1, 0] == expected


@pytest.mark.parametrize(
    "prices, lookback, days, crossover, expected",
    [
        (TEST_PRICES, 4, 1, True, 96 / ((96 + 94 + 95 + 98) / 4) - 1),
        (TEST_PRICES, 4, 2, True, 96 / ((96 + 94 + 95 + 98 + 97 + 98 + 99 + 101) / 8) - 1),
        (TEST_PRICES, 4, 2, False, (96 + 94 + 95 + 98 + 97 + 98 + 99 + 101) / 8),
    ],
)
def test_sma_cross_over(prices, lookback, days, crossover, expected):
    """Test simple moving average crossover signal of Signal class."""
    obj = Signal(prices).sma_crossover(lookback, crossover, days)
    assert obj.values[-1, 0] == expected


# TODO: create proper unit test for this
@pytest.mark.parametrize("prices, expected", [(TEST_PRICES, None)])
def test_protective_momentum_score(prices, expected, mocker):
    """Test protective momentum of Signal class."""
    assert 1 == 1
