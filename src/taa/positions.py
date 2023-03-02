"""Store base functions for weights and positions."""
from typing import List
from datetime import datetime

import numpy as np
import pandas as pd


def calculate_volatility(
    returns, lookback: int = 21, factor: int = 252, estimator: str = None
) -> pd.DataFrame:
    """Calculate historical volatility for all calendar days.

    Args:
        lookback (int, optional): window over which vol is calculated. Defaults to 21.
        factor (int, optional): number of business days in a year. Defaults to 252.
        estimator (str, optional): volatility estimator. Defaults to None.

    Returns:
        pd.DataFrame: time series of volatilities matching rebalance dates
    """
    rolling_volatility = returns.rolling(lookback).std() * np.sqrt(factor)
    return rolling_volatility.asfreq("D").ffill()


class Positions:
    """Base class for all portfolio weight classes."""

    def __init__(self, assets: List[str], rebalance_dates: List[datetime]):
        """Initialize base class with assets and rebalance dates.

        Creates dataframe ``self.weights`` which will be inherited downstream.

        Args:
            assets (_type_): list of tickers
            rebalance_dates (_type_): list of rebala
        """
        self.assets = assets
        self.rebalances_dates = rebalance_dates
        self.n_assets, self.n_obs = len(assets), len(rebalance_dates)

        # set up multilevel index
        index = pd.MultiIndex.from_product([rebalance_dates, assets])
        self.weights = pd.DataFrame(index=index)


class EqualWeights(Positions):
    """Store equally weighted portfolio."""

    def __init__(self, assets, rebalance_dates):
        """Initialize and create equally weighted portfolio."""
        super().__init__(assets, rebalance_dates)
        self.__name__ = "EW"
        position = np.ones(self.n_assets) / self.n_assets
        self.weights[self.__name__] = np.tile(position, self.n_obs)
        self.weights.sort_index(inplace=True)


# TODO: add more hist. volatility estimators and make kwargs explicit
class RiskParity(Positions):
    """Store naive implementation of Risk Parity."""

    def __init__(self, assets, rebalance_dates, returns: pd.DataFrame, **kwargs: dict):
        super().__init__(assets, rebalance_dates)
        self.returns = returns
        self.__name__ = "RP"

        # calculate historical vol (different estimators shall be used later)
        self.weights = self.create_weights(**kwargs)

    def create_weights(self, **kwargs: dict) -> pd.DataFrame:
        """Create risk parity weights and store them in dataframe.

        Args:
            **kwargs: keyword arguments for volatility estimation

        Returns:
            pd.DataFrame: weights for each asset
        """
        inverse_vols = 1 / calculate_volatility(self.returns, **kwargs)
        inverse_vols = inverse_vols.loc[self.rebalances_dates, :]
        vol_weights = inverse_vols.div(inverse_vols.sum(axis=1).values.reshape(-1, 1))
        return vol_weights.stack().rename(self.__name__).to_frame()
