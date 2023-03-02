# %%
%reload_ext autoreload
%autoreload 2

from src.taa.data import get_strategy_price_data
from src.taa.static import STRATEGIES
from src.taa.strategies import StrategyPipeline
from src.taa.signals import Signal
from src.taa.positions import EqualWeights, RiskParity
import pandas as pd

start, end = "2011-01-01", "2023-03-01"
# %%
data = get_strategy_price_data(STRATEGIES, start, end).dropna()

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

# %%
rebalance_dates = pd.bdate_range("2014-01-01", end, freq="BM")
assets = ["VTI", "VEU", "VNQ", "AGG"]
returns = data.pct_change().dropna().loc[:, assets]

ew = EqualWeights(assets, rebalance_dates)
rp = RiskParity(assets, rebalance_dates, returns)
print("Equal Weights\n", ew.weights)
print("Risk Parity \n", rp.weights)
