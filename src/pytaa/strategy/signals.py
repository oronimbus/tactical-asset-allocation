"""Computes signals used in the asset allocation schemes."""
import numpy as np
import pandas as pd


class Signal:
    """Momentum signal class used for tactical asset allocation."""

    def __init__(self, prices: pd.DataFrame):
        """Initialize signal class with daily returns.

        Args:
            prices (pd.DataFrame): table of daily returns
        """
        self.prices = prices
        self.monthly_prices = self.prices.resample("BM").last()

    def classic_momentum(self, start: int = 12, end: int = 1) -> pd.DataFrame:
        r"""Classic cross-sectional Momentum definition by Jegadeesh.

        The calculation follows:

        .. math::
            Z = \frac{P_{t-12}}{P_{t-1}} - 1

        For reference also see Asness (1994, 2013, 2014).

        Args:
            start (int, optional): beginning of momentum period. Defaults to 12.
            end (int, optional): end of momentum period. Defaults to 1.

        Returns:
            pd.DataFrame: table of momentum signal
        """
        momentum = self.monthly_prices.shift(end).div(self.monthly_prices.shift(start)) - 1
        return momentum

    def momentum_score(self) -> pd.DataFrame:
        """Calculate weighted average momentum for Vigilant portfolios."""
        score = np.zeros_like(self.monthly_prices)

        for horizon in [12, 4, 2, 1]:
            lag = int(12 / horizon)
            returns = self.monthly_prices.div(self.monthly_prices.shift(lag))
            score = score + (horizon * returns)

        norm_score = score - 19
        return norm_score

    def sma_crossover(
        self, lookback: int = 12, crossover: bool = True, days: int = 21
    ) -> pd.DataFrame:
        r"""Calculate simple Moving Average Crossover using monthly prices.

        Crossover score :math:`Z` over :math:`k` months is calculated as:

        .. math::

            Z = \frac{p_t}{k^{-1} \sum_{i=0}^{k} p_{t-i}} - 1

        If ``crossover`` is set to ``False`` then the simple moving average on its own is returned.

        Args:
            lookback (int, optional): number of months used for average. Defaults to 12.
            crossover (int, optional): returns crossover else just SMA. Defaults to True.
            days (int, optional): number of days in a business month. Defaults to 21.

        Returns:
            pd.DataFrame: table of signal strength
        """
        sma = self.prices.rolling(days * lookback).mean().resample("BM").last()
        if crossover:
            return self.monthly_prices.div(sma) - 1
        return sma

    def protective_momentum_score(self) -> pd.DataFrame:
        """Calculate momentum score for Generalized Protective Momentum portfolios."""
        returns = self.prices.pct_change()
        ew_basket = returns.mean(axis=1).rename("ew_basket")
        grouper = returns.join(ew_basket)
        rolling_corr = grouper.rolling(252).corr(grouper["ew_basket"]).resample("BM").last()

        avg_return = np.zeros_like(self.monthly_prices)
        for horizon in [1, 3, 6, 12]:
            avg_return += 0.25 * self.monthly_prices.div(self.monthly_prices.shift(horizon))

        score = avg_return * (1 - rolling_corr)
        return score
