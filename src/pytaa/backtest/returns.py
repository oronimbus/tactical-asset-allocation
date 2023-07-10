"""Calculate portfolio strategy returns."""
import numba
import numpy as np
import pandas as pd

from pytaa.tools.data import (
    get_currency_returns,
    get_historical_dividends,
    get_historical_price_data,
    get_issue_currency_for_tickers,
)

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


@numba.njit
def calculate_drifted_weight_returns(
    returns: np.array, weights: np.array, rebal_index: np.array
) -> np.array:
    r"""Project cumulative daily returns onto lower frequency returns

    The portfolio weights are iteratively updated using market performance:
    
    .. math::
    
        w_{i,t} = \frac{w_{i,t-1} \times (1 + r_{i,t})}{\sum_j^N{w_{j,t-1} \times (1 + r_{j,t})}}\\
        r_{p,t} = \sum_i^N w_{i,t} \times r_{i,t}\\

    Args:
        weights (np.array): KxM table of portfolio weights
        returns (np.array): NxM table of daily returns
        rebal_index (np.array): Kx0 1d array of rebalance indices for returns
    Returns:
        np.array: Nx1 vector of drifted returns
    """
    n_obs, n_assets = returns.shape[0], weights.shape[1]
    total_return = np.zeros((n_obs, 1), dtype=np.float64)
    w_drift = np.zeros(n_assets)

    for i in range(n_obs):
        if i in rebal_index:
            w_drift = weights[np.where(rebal_index == i)[0][0], :]
        else:
            r_day = returns[i, :]
            total_return[i, :] = np.nansum(w_drift * r_day)
            w_drift = (w_drift * (1 + r_day)) / np.nansum(w_drift * (1 + r_day))
    return total_return


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

    def _process_strategy(self, returns: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Project weights onto returns for a given strategy.

        Args:
            returns (pd.DataFrame): table of asset returns
            strategy (str): label of strategy

        Returns:
            pd.DataFrame: table of simple returns per strategy
        """
        strat_weights = self.weights[strategy].unstack().fillna(0)
        strat_returns = returns.loc[:, strat_weights.columns]
        temp_idx = pd.bdate_range(strat_weights.index.min(), strat_returns.index.max())
        strat_returns = strat_returns.reindex(temp_idx).fillna(0)
        rebal_index = list(map(strat_returns.index.get_loc, strat_weights.index))
        total_return = calculate_drifted_weight_returns(
            np.asarray(strat_returns), np.asarray(strat_weights), np.asarray(rebal_index)
        )
        return pd.DataFrame(total_return, index=strat_returns.index, columns=[strategy])

    def run(
        self, end_date: str = None, frequency: str = None, return_type: str = "total"
    ) -> pd.DataFrame:
        """Run backtester and return strategy returns.

        Args:
            end_date (str, optional): end date of strategy. Defaults to None.
            frequency (str, optional): frequency of portfolio returns. Defaults to None.
            return_type (str, optional): either ``price`` or ``total`` return
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
        returns = get_historical_total_return(prices, self.portfolio_currency, return_type)
        portfolio_total_return = []

        for strategy in self.weights.columns:
            total_return = self._process_strategy(returns, strategy)
            portfolio_total_return.append(total_return)

        # concat all strategy returns, then aggregate to chosen frequency
        portfolio_total_return = pd.concat(portfolio_total_return, axis=1)
        portfolio_total_return.index.name = "Date"

        if frequency is None:
            resampled = portfolio_total_return.groupby(pd.Grouper(freq=self.frequency))
            return resampled.apply(lambda x: (1 + x).prod() - 1)
        elif frequency == "D":
            return portfolio_total_return
        raise NotImplementedError
