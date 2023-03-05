"""Data handlers."""
from typing import Union, List
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from src.taa.strategy.strategies import StrategyPipeline


def validate_tickers(tickers: List[str], raise_issue: bool = False) -> bool:
    """Validate list of security tickers.

    Args:
        tickers (List[str]): list of tickers to validate
        raise_issue (bool, optional): raises error if set to true. Defaults to False.

    Returns:
        bool: False if validation failed and else True

    Raises:
        Exception: if tickers are invalid and ``raise_issue`` has been set to True
    """
    is_valid = True
    if len(tickers) < 1 or any(x is None for x in tickers):
        is_valid = False

    if not is_valid and not raise_issue:
        return is_valid
    elif not is_valid and raise_issue:
        raise Exception("Tickers are not valid in Yahoo! Finace. Review inputs before proceeding.")
    return True


def get_historical_price_data(
    tickers: List[str], start_date: str, end_date: str, **kwargs: dict
) -> pd.DataFrame:
    """Simple data request using Yahoo! Finance API.

    Args:
        tickers (List[str]): list of tickers
        start_date (str): start date of time series
        end_date (str): end date of time series

    Returns:
        pd.DataFrame: table of prices
    """
    tickers = list(set(tickers))
    is_valid = validate_tickers(tickers, **kwargs)
    if not is_valid:
        return pd.DataFrame(columns=["Close", "Open", "Low", "High", "Volume", "Adj Close"])

    # if tickers are valid then launch yfinance API
    str_of_tickers = " ".join(tickers)
    data = yf.download(str_of_tickers, start=start_date, end=end_date, progress=False)
    if data.columns.nlevels <= 1:
        data.columns = pd.MultiIndex.from_product([data.columns, tickers])
    return data


def get_currency_returns(
    currency_list: List[str],
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    base_currency: str = "USD",
    **kwargs: dict,
) -> pd.DataFrame:
    """Retrieve currency returns versus base.

    Currencies are quoted as CCY1/CCY2 where CCY1 refers to the base currency. For ease, FX
    conventions are disregarded. For example, EUR/USD would be retrieved as USD/EUR meaning long
    USD and short EUR. Fills missing values with zeros. FX returns are returned in the same order
    as passed in.

    Args:
        currency_list (List[str]): list of currency tickers, e.g. "EUR"
        start_date (Union[str, datetime]): start date of time series
        end_date (Union[str, datetime]): end date of time series
        base_currency (str, optional): quotation basis (long). Defaults to "USD".

    Returns:
        pd.DataFrame: table fo currency returns
    """
    to_process = [f"{base_currency}{fx}=X" for fx in currency_list if fx != base_currency]
    price_data = get_historical_price_data(to_process, start_date, end_date, **kwargs)

    fx_returns = price_data.loc[:, "Adj Close"].pct_change().dropna()
    date_range = pd.bdate_range(start_date, end_date)
    zeros = pd.Series(np.zeros(date_range.shape[0]), index=date_range)
    to_concat = []

    for fx in currency_list:
        if fx == base_currency:
            to_concat.append(zeros)
        else:
            to_concat.append(fx_returns.loc[:, f"{base_currency}{fx}=X"])
    fx_returns = pd.concat(to_concat, axis=1).fillna(0)
    fx_returns.columns = currency_list
    return fx_returns


def get_issue_currency_for_tickers(tickers: List[str]) -> List[str]:
    """Retrieve currency in which security was issued in.

    Args:
        tickers (List[str]): list of tickers

    Returns:
        List[str]: list of currency tickers, e.g. "EUR"
    """
    return ["USD" for _ in tickers]


def get_historical_total_return(
    price_data: pd.DataFrame, portfolio_currency: str = None, return_type: str = "price"
) -> pd.DataFrame:
    """Calculate daily total return in portfolio currency.

    Args:
        price_data (pd.DataFrame): table of closing price
        portfolio_currency (str, optional): portfolio currency for returns. Defaults to None.
        return_type (str, optional): returns gross or net of dividends. Defaults to "price".

    Returns:
        pd.DataFrame: table of asset returns

    Raises:
        NotImplementedError: if return_type is not equal to "price"
    """
    if return_type == "price":
        returns = price_data.pct_change().dropna()
    else:
        raise NotImplementedError

    if portfolio_currency is None:
        return returns

    # handle currency adjustment (not proven to work yet!)
    currencies = get_issue_currency_for_tickers(price_data.columns.to_list())
    if all([fx == portfolio_currency for fx in currencies]):
        return returns

    start_date, end_date = price_data.index.min(), price_data.index.max()
    fx_returns = get_currency_returns(currencies, start_date, end_date, portfolio_currency)
    fx_returns = fx_returns.reindex(returns.index).fillna(0)
    fx_adj_returns = returns - fx_returns.values
    return fx_adj_returns


def get_strategy_price_data(
    pipeline: StrategyPipeline,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime] = None,
) -> pd.DataFrame:
    """Request and store strategy data in dataframe.

    Args:
        pipeline (StrategyPipeline): strategy pipeline
        start_date (Union[str, datetime]): start date of strategies
        end_date (Union[str, datetime], optional): end date of strategies (None equals today). \
            Defaults to None.

    Returns:
        pd.DataFrame: table of adjusted prices for all strategy inputs
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    all_tickers = []
    for strategy in pipeline.pipeline:
        all_tickers += strategy.get_tickers()

    data = get_historical_price_data(all_tickers, start_date, end_date)
    return data.loc[:, "Adj Close"]


if __name__ == "__main__":
    from src.taa.strategy.static import STRATEGIES
    from src.taa.strategy.strategies import StrategyPipeline

    pipeline = StrategyPipeline(STRATEGIES)
    print(get_strategy_price_data(pipeline, "2011-01-01").dropna())
