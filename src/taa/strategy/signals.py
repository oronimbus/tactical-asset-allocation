"""Computes signals used in the asset allocation schemes."""
import pandas as pd
import numpy as np


class Signal:
    def __init__(self, prices: pd.DataFrame):
        """Initialize signal class with daily returns.

        Args:
            prices (pd.DataFrame): table of daily returns
        """
        self.prices = prices
        self.monthly_prices = self.prices.resample("BM").last()

    def momentum_score(self) -> pd.DataFrame:
        """Used in Vigilant portfolios."""
        score = np.zeros_like(self.monthly_prices)

        for horizon in [12, 4, 2, 1]:
            lag = int(12 / horizon)
            returns = self.monthly_prices.div(self.monthly_prices.shift(lag))
            score += horizon * returns

        norm_score = score - 19
        return norm_score

    def sma_crossover(self, lookback: int = 12) -> pd.DataFrame:
        r"""Simple Moving Average Crossover using monthly prices.

        Crossover score :math:`Z` over :math:`k` months is calculated as:

        .. math::

            Z = \frac{p_t}{k^{-1} \sum_{i=0}^{k} p_{t-i}} - 1

        Args:
            lookback (int, optional): number of months used for average. Defaults to 12.

        Returns:
            pd.DataFrame: table of signal strength
        """
        sma = self.prices.rolling(21 * lookback).mean().resample("BM").last()
        cross_over = self.monthly_prices.div(sma) - 1
        return cross_over

    def protective_momentum_score(self):
        """User in Generalized Protective Momentum."""
        returns = self.prices.pct_change()
        ew_basket = returns.mean(axis=1).rename("ew_basket")
        grouper = returns.join(ew_basket)
        rolling_corr = grouper.rolling(252).corr(grouper["ew_basket"]).resample("BM").last()

        avg_return = np.zeros_like(self.monthly_prices)
        for horizon in [1, 3, 6, 12]:
            avg_return += 0.25 * self.monthly_prices.div(self.monthly_prices.shift(horizon))

        score = avg_return * (1 - rolling_corr)
        return score
