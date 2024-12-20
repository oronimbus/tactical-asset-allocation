"""Store data containers for strategies."""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class Strategy:
    """Container class for strategies."""

    tag: str
    name: str
    risk_assets: List[str] = field(default=None)
    safe_assets: List[str] = field(default=None)
    canary_assets: List[str] = field(default=None)
    asset_classes: List[str] = field(default=None)
    weights: List[float] = field(default=None)
    frequency: str = field(default="M")

    def get_tickers(self):
        """Return list of tickers used in strategy."""
        tickers = []
        for assets in [self.risk_assets, self.safe_assets, self.canary_assets]:
            tickers += assets if assets is not None else []
        return list(set(tickers))


class StrategyPipeline:
    """Pipeline for strategy containers."""

    def __init__(self, strategies: List[dict]):
        """Initialize strategy pipeline class."""
        self.pipeline = self._parse_strategies(strategies)

    def _parse_strategies(self, strategies: List[dict]):
        """Parse strategy list and set attribute to tag."""
        pipeline = ()
        for strategy in strategies:
            strat_obj = Strategy(
                tag=strategy["tag"],
                name=strategy["name"],
                risk_assets=strategy.get("riskAssets"),
                safe_assets=strategy.get("safeAssets"),
                canary_assets=strategy.get("canaryAssets"),
                asset_classes=strategy.get("assetClasses"),
                weights=strategy.get("weights"),
                frequency=strategy.get("frequency"),
            )
            setattr(self, strategy["tag"].lower(), strat_obj)
            pipeline += (strat_obj,)

        return pipeline
