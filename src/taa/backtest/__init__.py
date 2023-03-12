"""Feature backtest modules."""
from src.taa.backtest.performance import Tearsheet
from src.taa.backtest.positions import Positions, EqualWeights, RiskParity
from src.taa.backtest.returns import (
    get_historical_total_return,
    get_historical_price_data,
    Backtester,
)


__all__ = [
    "Tearsheet",
    "Positions",
    "EqualWeights",
    "RiskParity",
    "get_historical_total_return",
    "get_historical_price_data",
    "Backtester",
]
