"""Store base functions for weights and positions."""
from datetime import datetime
from typing import Callable, List

import numpy as np
import pandas as pd

from pytaa.tools.risk import (
    calculate_risk_parity_portfolio,
    calculate_min_variance_portfolio,
    calculate_rolling_volatility,
    Covariance,
)


class Positions:
    """Base class for all portfolio weight classes."""

    def __init__(self, assets: List[str], rebalance_dates: List[datetime]):
        """Initialize base class with assets and rebalance dates.

        Creates dataframe ``self.weights`` which will be inherited downstream.

        Args:
            assets (List[str]): list of tickers
            rebalance_dates (List[str]): list of rebalance dates
        """
        self.assets = assets
        self.rebalance_dates = rebalance_dates
        self.n_assets, self.n_obs = len(assets), len(rebalance_dates)

        # set up multilevel index
        index = pd.MultiIndex.from_product([rebalance_dates, assets])
        self.weights = pd.DataFrame(index=index)


class EqualWeights(Positions):
    """Store equally weighted portfolio."""

    def __init__(self, assets: List[str], rebalance_dates: List[datetime]):
        """Initialize and create equally weighted portfolio."""
        super().__init__(assets, rebalance_dates)
        self.__name__ = "EW"
        position = np.ones(self.n_assets) / self.n_assets
        self.weights[self.__name__] = np.tile(position, self.n_obs)
        self.weights.sort_index(inplace=True)
        self.weights.index.names = ["Date", "ID"]


class RiskParity(Positions):
    """Store naive implementation of Risk Parity."""

    def __init__(
        self,
        assets: List[str],
        rebalance_dates: List[datetime],
        returns: pd.DataFrame,
        estimator: str = "hist",
        lookback: int = 21,
        **kwargs: dict
    ):
        """Initialize class for Risk Parity calculation.

        Args:
            assets (List[str]): list of asset tickers
            rebalance_dates (List[datetime]): list of rebalancing dates
            returns (pd.DataFrame): dataframe of daily asset returns
            estimator (str, optional): volatility estimation method. Defaults to "hist".
            lookback (float, optional): volatility estimation window. Defaults to 21.
        """
        super().__init__(assets, rebalance_dates)
        self.returns = returns
        self.__name__ = "RP"

        # calculate historical vol (different estimators shall be used later)
        self.weights = self.create_weights(estimator, lookback, **kwargs)
        self.weights.index.names = ["Date", "ID"]

    def create_weights(self, method: str, lookback: float, **kwargs: dict) -> pd.DataFrame:
        """Create risk parity weights and store them in dataframe.

        The estimator can be one of: ``ewma``, ``hist``, ``equal_risk`` or ``min_variance``. The
        latter two involve an optimization process.

        Additional estimation parameters can be passed as keyword arguments. For example you can
        pass the halflife parameter for ``alpha=0.94`` when using ``ewma``.

        Args:
            method (str, optional): volatility estimation method.
            lookback (float, optional): volatility estimation window.
            **kwargs: keyword arguments for volatility estimation

        Returns:
            pd.DataFrame: weights for each asset
        """
        if method in ["hist", "ewma"]:
            inverse_vols = 1 / calculate_rolling_volatility(
                self.returns, estimator=method, lookback=lookback, **kwargs
            )
            inverse_vols = inverse_vols.reindex(self.rebalance_dates)
            weights = inverse_vols.div(inverse_vols.sum(axis=1).values.reshape(-1, 1))

        elif method in ["equal_risk", "min_variance"]:
            optimizer = {
                "min_variance": calculate_min_variance_portfolio,
                "equal_risk": calculate_risk_parity_portfolio,
            }
            weights = rolling_optimization(
                self.returns, self.rebalance_dates, optimizer[method], lookback, **kwargs
            )
        else:
            raise NotImplementedError

        weights = np.maximum(0, weights)
        return weights.stack().rename(self.__name__).to_frame()


def rolling_optimization(
    returns: pd.DataFrame,
    rebalance_dates: List[datetime],
    optimizer: Callable,
    lookback: int,
    shrinkage: str = None,
    shrinkage_factor: float = None,
) -> pd.DataFrame:
    """Perform rolling optimization over rebalance dates.

    Args:
        returns (pd.DataFrame): dataframe of daily asset returns
        rebalance_dates (List[datetime]): list of rebalancing dates
        optimizer (Callable): optimization routine, e.g. ``calculate_risk_parity_portfolio``
        lookback (int): window for covariance data
        shrinkage (str, optional): covariance shrinkage method. Defaults to None.
        shrinkage_factor (float, optional): covariance shrinkage factor. Defaults to None.

    Returns:
        pd.DataFrame: table of optimized asset weights
    """
    weights = []

    for date in rebalance_dates:
        data = returns[returns.index <= date].iloc[-lookback:, :]
        cov = Covariance(data, shrinkage, shrinkage_factor)

        # this is not done yet, just a skeleton of what will work, eventually
        w_opt = optimizer(cov)
        weights.append(w_opt)

    weights = pd.DataFrame(np.row_stack(weights), index=rebalance_dates, columns=returns.columns)
    return weights


def vigilant_allocation(
    data: pd.Series,
    risk_assets: List[str],
    safe_assets: List[str],
    top_k: int = 5,
    step: float = 0.25,
) -> pd.DataFrame:
    """Allocate assets based on threshold using scores.

    Used in computing the Vigilant portfolios. The allocation works as follows (using $k=5$):
    Determine the number of assets $n$ with negative $Z$, if $n>4$ allocate 100% in safe asset with
    highest momentum score, if $n=3$ put 75% in safest asset, remaining 25% is split equally in 5
    risk assets with highest momentum, if $n=2$ put 50% in safest asset, 50% split evenly top 5
    risk assets etc.

    Args:
        data (pd.Series): dataframe with signals
        risk_assets (List[str]): list of risky assets
        safe_assets (List[str]): list of safety assets
        top_k (int, optional): rank threshold. Defaults to 5.
        step (float, optional): step in allocation to risk assets given signal. Defaults to 0.25.

    Returns:
        pd.DataFrame: dataframe of weights
    """
    is_neg = sum(np.where(data < 0, 1, 0))
    empty = data * np.nan
    safety = pd.concat([data.loc[safe_assets].rank(ascending=False), empty.loc[risk_assets]])
    safety = safety[~safety.index.duplicated()].sort_index()
    risky = pd.concat([data.loc[risk_assets].rank(ascending=False), empty.loc[safe_assets]])
    risky = risky[~risky.index.duplicated()].sort_index()

    # allocate assets based on number of negative scores
    safe_weights = np.where(safety == 1, min([1, step * is_neg]), 0)
    risk_weights = np.where(risky <= top_k, (1 - min([1, step * is_neg])) / top_k, 0)
    weights = safe_weights + risk_weights
    return pd.DataFrame(weights, index=safety.index).T
