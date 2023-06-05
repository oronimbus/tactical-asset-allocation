import pytest

from pytaa.strategy.strategies import Strategy, StrategyPipeline


@pytest.mark.parametrize("tag, name, ra, sa, ca, ac, wgts, freq, expected", [
    ("A", "abc", [1, 2, 2, 3, 3], [4], [], [], [], "M", [1,2,3,4])
])
def test_strategy(tag, name, ra, sa, ca, ac, wgts, freq, expected):
    """Test main Strategy class."""
    obj = Strategy(tag, name, ra, sa, ca, ac, wgts, freq)
    assert obj.get_tickers() == expected
    

@pytest.mark.parametrize("strats, expected", [
    ([{"tag": "A", "name": "abc"}, {"tag": "B", "name": "def"}], 2)
])
def test_strategy_pipeline(strats, expected):
    """Test StrategyPipeline class."""
    pipe = StrategyPipeline(strats).pipeline
    assert len(pipe) == expected
