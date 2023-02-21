"""Data handlers."""
from typing import Union, List
from datetime import datetime

import pandas as pd
import yfinance as yf
from taa.strategies import STRATEGIES


def get_strategy_price_data(
    strategies: List[dict], start_date: Union[str, datetime], end_date: Union[str, datetime] = None
) -> pd.DataFrame:
    """Store strategy data in dataframe."""
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    all_tickers = []
    for strategy in strategies:
        for assets in ["riskAssets", "safeAssets", "canaryAssets"]:
            all_tickers += strategy.get(assets, [])

    str_of_tickers = " ".join(set(all_tickers))
    data = yf.download(str_of_tickers, start=start_date, end=end_date)
    return data.loc[:, "Adj Close"]


if __name__ == "__main__":
    print(get_strategy_price_data(STRATEGIES, "2011-01-01").dropna())
