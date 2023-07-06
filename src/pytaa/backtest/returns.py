"""Calculate portfolio strategy returns."""
import numpy as np
import pandas as pd

from pytaa.tools.data import (get_currency_returns, get_historical_dividends,
                              get_historical_price_data,
                              get_issue_currency_for_tickers)

pd.options.mode.chained_assignment = None


def get_historical_total_return(
    price_data: pd.DataFrame, portfolio_currency: str = None, return_type: str = "total"
) -> pd.DataFrame:
    r"""Calculate daily total return in portfolio currency.

    The one day total return :math:`r_t` is calculated as:

    .. math::

        r_{t,t-1}=\frac{p_t + d_t - p_{t-1}}{p_{t-1}}

    The dividends :math:`d_t` are retrieved from Yahoo! Finance. We use the close price instead
    of the adjusted close price. Both are stock split adjusted but the adjusted close is also
    backwards-adjusted for dividends, i.e. the dividend paid is subtract from the denominator in
    the return calculation.

    Args:
        price_data (pd.DataFrame): table of closing price
        portfolio_currency (str, optional): portfolio currency for returns. Defaults to None.
        return_type (str, optional): returns gross or net of dividends. Defaults to "total".

    Returns:
        pd.DataFrame: table of asset returns

    Raises:
        NotImplementedError: if return_type is not equal to "price"
    """
    start_date, end_date = price_data.index.min(), price_data.index.max()
    tickers = price_data.columns.to_list()

    if return_type == "price":
        returns = price_data.pct_change().dropna()
    elif return_type == "total":
        dividends = get_historical_dividends(tickers, start_date, end_date)
        dividends = dividends.reindex(price_data.index).fillna(0).loc[:, tickers]
        returns = price_data.add(dividends).div(price_data.shift(1).values).sub(1).dropna()
    else:
        raise NotImplementedError

    if portfolio_currency is None:
        return returns

    # handle currency adjustment (not proven to work yet!)
    currencies = get_issue_currency_for_tickers(tickers)
    if all([fx == portfolio_currency for fx in currencies]):
        return returns

    fx_returns = get_currency_returns(currencies, start_date, end_date, portfolio_currency)
    fx_returns = fx_returns.reindex(returns.index).fillna(0)
    fx_adj_returns = returns - fx_returns.values
    return fx_adj_returns


def calculate_drifted_weight_returns(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    r"""Project cumulative daily returns onto lower frequency returns

    The portfolio weights are iteratively updated using market performance:
    
    .. math::
    
        w_{i,t} = \frac{w_{i,t-1} \times (1 + r_{i,t})}{\sum_j^N{w_{j,t-1} \times (1 + r_{j,t})}}\\
        r_{p,t} = \sum_i^N w_{i,t} \times r_{i,t}\\

    Args:
        weights (pd.DataFrame): table of portfolio weights
        returns (pd.DataFrame): table of daily returns

    Returns:
        pd.DataFrame: table of drifted weights
    """
    total_return = []
    for start, end in zip(weights.index[:-1], weights.index[1:]):
        w_drift = weights.loc[start, :]
        simple_returns = returns[(returns.index > start) & (returns.index <= end)]
        n_obs = simple_returns.shape[0]
        period_returns = np.empty(n_obs)

        for i in range(n_obs):
            r_day = np.array(simple_returns)[i, :]
            period_returns[i] = np.nansum(w_drift * r_day)
            w_drift = (w_drift * (1 + r_day)) / np.nansum(w_drift * (1 + r_day))

        total_return.append(pd.Series(period_returns, index=simple_returns.index))
    return pd.concat(total_return)


class Backtester:
    def __init__(self, weights: pd.DataFrame, portfolio_currency: str = "USD", **kwargs):
        """Initialize backtester with weights matrix.

        Args:
            weights (pd.DataFrame): multi-level dataframe with weights
            portfolio_currency (str): currency in which returns are calculated. Defaults to USD.
        """
        self.weights = weights
        self.portfolio_currency = portfolio_currency
        self.rebal_dates = self.weights.index.get_level_values(0).unique()
        self.assets = self.weights.index.get_level_values(1).unique()
        self.strategies = self.weights.columns
        self.frequency = pd.infer_freq(self.rebal_dates)

    def _process_strategy(
        self, weights: pd.DataFrame, returns: pd.DataFrame, strategy: str
    ) -> pd.DataFrame:
        """Project weights onto returns for a given strategy.

        Args:
            weights (pd.DataFrame): table of portfolio weights
            returns (pd.DataFrame): table of asset returns
            strategy (str): label of strategy

        Returns:
            pd.DataFrame: table of simple returns per strategy
        """
        # prep input data for right shape
        weights = weights.loc[:, [strategy]].unstack().droplevel(0, 1)
        returns.loc[weights.index.min(), :] = 0
        returns.sort_index(inplace=True)

        # calculate drifted weights, then project weights onto returns
        strategy_returns = calculate_drifted_weight_returns(weights, returns[weights.columns])
        return strategy_returns.rename(strategy).to_frame()

    def run(self, end_date: str = None, frequency: str = None, **kwargs) -> pd.DataFrame:
        """Run backtester and return strategy returns.

        Args:
            end_date (str, optional): end date of strategy. Defaults to None.
            frequency (str, optional): frequency of portfolio returns. Defaults to None.

        Returns:
            pd.DataFrame: table of portfolio returns

        Raises:
            NotImplementedError: if frequency is neither None or "D"
        """
        start_date = self.rebal_dates.min() - pd.offsets.BDay(1)
        if end_date is None:
            end_date = self.rebal_dates.max() + pd.offsets.BDay(1)

        # retrieve data for total return calculation
        prices = get_historical_price_data(self.assets, start_date, end_date).loc[:, "Close"]
        returns = get_historical_total_return(prices, self.portfolio_currency, **kwargs)
        portfolio_total_return = []

        for start, end in zip(self.rebal_dates[:-1], self.rebal_dates[1:]):
            period_returns = returns[(returns.index > start) & (returns.index <= end)]
            period_weights = self.weights[
                (self.weights.index.get_level_values(0) >= start)
                & (self.weights.index.get_level_values(0) <= end)
            ]

            # process each strategy separately
            period_results = []
            for strategy in self.strategies:
                strategy_returns = self._process_strategy(period_weights, period_returns, strategy)
                period_results.append(strategy_returns)

            portfolio_total_return.append(pd.concat(period_results, axis=1))

        # concat all strategy returns, then aggregate to chosen frequency
        portfolio_total_return = pd.concat(portfolio_total_return).sort_index()

        if frequency is None:
            resampled = portfolio_total_return.groupby(pd.Grouper(freq=self.frequency))
            return resampled.apply(lambda x: (1 + x).prod() - 1)
        elif frequency == "D":
            return portfolio_total_return
        raise NotImplementedError
