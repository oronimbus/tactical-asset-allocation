"""Data handling and webscraping tools."""
from typing import Union, List
from datetime import datetime
from lxml import html
from lxml.etree import tostring
import requests

import numpy as np
import pandas as pd
import yfinance as yf

from taa.strategy.strategies import StrategyPipeline


def get_historical_dividends(
    tickers: List[str], start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    """Retrieve historical dividends for universe of stocks using Yahoo! Finance.

    Args:
        tickers (List[str]): list of equity tickers
        start_date (datetime): start date of series
        end_date (datetime): end date of series

    Returns:
        pd.DataFrame: table of dividends
    """
    dividends = [yf.Ticker(x).dividends.rename(x).to_frame() for x in tickers]
    div_table = pd.concat(dividends, axis=1)
    return div_table.loc[(div_table.index >= start_date) & (div_table.index <= end_date)]


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

    fx_returns = price_data.loc[:, "Close"].pct_change().dropna()
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


def find_ticker_currency(ticker: str, fallback: str = "USD") -> str:
    """Find currency of ticker using Yahoo! Finance website.

    If the currency cannot be found then it is assumed to be USD (or which ever fallback specified).
    Each request times out after 30 seconds.

    Args:
        ticker (str): ticker symbol

    Returns:
        str: 3 character string of currency code, e.g. ``USD``
    """
    url = f"http://finance.yahoo.com/quote/{ticker}?p={ticker}"
    
    try:
        response = requests.get(url, timeout=30)
        parser = html.fromstring(response.text)
        inner_html = str(tostring(parser))
        to_find = "Currency in"
        loc_ccy = inner_html.find(to_find) + len(to_find) + 1
        return inner_html[loc_ccy : loc_ccy + 3]
    except:
        return fallback


def get_issue_currency_for_tickers(tickers: List[str], assumed_currency: str = None) -> List[str]:
    """Retrieve currency in which security was issued in.

    If the assumed currency is anything but None, then no lookup will be performed and the assumed
    currency is returned for all tickers. This is to speed up the request since scraping from
    Yahoo! Finance can be slow. For example a request of 5 tickers takes around 3-4 seconds.

    Args:
        tickers (List[str]): list of tickers

    Returns:
        List[str]: list of currency tickers, e.g. ``EUR``
    """
    if assumed_currency is not None:
        return [assumed_currency for _ in tickers]
    return [find_ticker_currency(ticker) for ticker in tickers]


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
    return data.loc[:, "Close"]


if __name__ == "__main__":
    from src.taa.strategy.static import STRATEGIES
    from src.taa.strategy.strategies import StrategyPipeline

    pipeline = StrategyPipeline(STRATEGIES)
    print(get_strategy_price_data(pipeline, "2011-01-01").dropna())
