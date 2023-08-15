"""Data handling and webscraping tools."""
import concurrent.futures
import logging
from datetime import datetime
from typing import List, Union

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from lxml import html
from lxml.etree import tostring

from pytaa.strategy.static import VALID_CURRENCIES
from pytaa.strategy.strategies import StrategyPipeline
from pytaa.tools.logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


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
    div_table.index = div_table.index.tz_localize(None)
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
    if not is_valid and raise_issue:
        raise Exception("Tickers are not valid in Yahoo! Finace. Review inputs before proceeding.")
    return True


def get_historical_price_data(
    tickers: List[str], start_date: str, end_date: str, **kwargs: dict
) -> pd.DataFrame:
    """Request price data using Yahoo! Finance API.

    Additional keyword arguments can be passed to control if and how tickers are validated. You
    can pass ``raise_issue=True`` to raise an Exception if an invalid ticker is present.

    Args:
        tickers (List[str]): list of tickers
        start_date (str): start date of time series
        end_date (str): end date of time series
        **kwargs (dict): key word arguments for ticker validation

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
    for ccy in currency_list:
        if ccy == base_currency:
            to_concat.append(zeros)
        else:
            to_concat.append(fx_returns.loc[:, f"{base_currency}{ccy}=X"])

    fx_returns = pd.concat(to_concat, axis=1).fillna(0)
    fx_returns.columns = currency_list
    return fx_returns


def find_ticker_currency(ticker: str, fallback: str = "USD") -> str:
    """Find currency of ticker using Yahoo! Finance website.

    If the currency cannot be found then it is assumed to be USD (or which ever fallback specified).
    Each request times out after 10 seconds.

    Args:
        ticker (str): ticker symbol

    Returns:
        str: 3 character string of currency code, e.g. ``USD``
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
            AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36",
    }

    url = f"http://finance.yahoo.com/quote/{ticker}?p={ticker}"

    try:
        response = requests.get(url, headers=headers, timeout=10)
        parser = html.fromstring(response.text)
        inner_html = str(tostring(parser))
        to_find = "Currency in"
        loc_ccy = inner_html.find(to_find) + len(to_find) + 1
        currency = inner_html[loc_ccy: loc_ccy + 3]
        if currency.upper() in VALID_CURRENCIES:
            return currency

        # if we get here it means the currency is not valid and we log a warning
        msg = f"Unsucessful GET request for {ticker}. Using default currency {fallback}."
        logger.warning(msg)
        return fallback
    except Exception as e:
        logger.warning("Error in request: %s. Returning fallback currency.", e)
        return fallback


def get_issue_currency_for_tickers(tickers: List[str]) -> List[str]:
    """Retrieve currency in which security was issued in.

    A request of 5 tickers takes around 3-4 seconds. The request is sped up using asynchronous
    execution.

    Args:
        tickers (List[str]): list of tickers

    Returns:
        List[str]: list of currency tickers, e.g. ``EUR``
    """
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(find_ticker_currency, ticker): ticker for ticker in tickers}
        for future in concurrent.futures.as_completed(futures):
            try:
                results[futures[future]] = future.result()
            except Exception as exc:
                logger.warning(exc)
    return [results[ticker] for ticker in tickers]


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
    from src.pytaa.strategy.static import STRATEGIES

    some_pipeline = StrategyPipeline(STRATEGIES)
    print(get_strategy_price_data(some_pipeline, "2011-01-01").dropna())
