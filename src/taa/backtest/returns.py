"""Calculate portfolio strategy returns."""
import pandas as pd

from src.taa.tools.data import get_historical_price_data


class Backtester:
    def __init__(self, weights: pd.DataFrame, **kwargs):
        """Initialize backtester with weights matrix.

        Args:
            weights (pd.DataFrame): multi-level dataframe with weights
        """
        self.weights = weights
        self.rebal_dates = self.weights.index.get_level_values(0).unique()
        self.assets = self.weights.index.get_level_values(1).unique()

    def run(self, end_date: str = None, **kwargs) -> pd.DataFrame:
        """Run backtester and return strategy returns.

        Args:
            end_date (str, optional): end date of strategy. Defaults to None.

        Returns:
            pd.DataFrame: table of portfolio returns
        """
        start_date = self.rebal_dates.min() - pd.offsets.BDay(1)
        if end_date is None:
            end_date = self.rebal_dates.max() + pd.offsets.BDay(1)
        price_data = get_historical_price_data(self.assets, start_date, end_date)
        returns = price_data.loc[:, "Adj Close"].pct_change().dropna()
        return returns
