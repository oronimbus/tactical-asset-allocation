"""Data handlers."""
from typing import Union, List
from datetime import datetime

import pandas as pd
import yfinance as yf

from src.taa.strategy.strategies import StrategyPipeline


def get_historical_price_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Simple data request using Yahoo! Finance API.

    Args:
        tickers (List[str]): list of tickers
        start_date (str): start date of time series
        end_date (str): end date of time series

    Returns:
        pd.DataFrame: table of prices
    """
    str_of_tickers = " ".join(set(tickers))
    data = yf.download(str_of_tickers, start=start_date, end=end_date, progress=False)
    return data


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
