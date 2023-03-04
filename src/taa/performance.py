"""Feature post-trade tools to analyse strategy performance."""
import pandas as pd


class Tearsheet:
    def __init__(self, returns: pd.DataFrame, benchmark: pd.DataFrame = None):
        """Initialize tearsheet with strategy returns and benchmark.

        Args:
            returns (pd.DataFrame): table of strategy returns
            benchmark (pd.DataFrame, optional): table of benchmark returns. Defaults to None.
        """
        self.returns = returns
        self.benchmark = benchmark

    def summary(self) -> pd.DataFrame:
        pass
