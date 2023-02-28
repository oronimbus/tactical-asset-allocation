from src.taa.data import get_strategy_price_data
from src.taa.static import STRATEGIES
from src.taa.strategies import StrategyPipeline
from src.taa.signals import Signal
data = get_strategy_price_data(STRATEGIES, "2011-01-01").dropna()

# test signals
signal_1 = Signal(data).momentum_score()
print(signal_1)
signal_2 = Signal(data).sma_crossover()
print(signal_2)
signal_3 = Signal(data).protective_momentum_score()
print(signal_3)

# test pipeline
pipeline = StrategyPipeline(STRATEGIES)
print(pipeline.ivy)
print(pipeline.kdaaa)