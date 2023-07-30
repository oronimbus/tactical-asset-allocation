"""Feature post-trade tools to analyse strategy performance."""
from collections import OrderedDict

import numpy as np
import pandas as pd

from pytaa.tools.risk import autocorrelation


# TODO: verify number, formulas taken from memory
class Tearsheet:
    def __init__(self, returns: pd.DataFrame, benchmark: pd.DataFrame = None):
        """Initialize tearsheet with strategy returns and benchmark.

        Args:
            returns (pd.DataFrame): table of strategy returns
            benchmark (pd.DataFrame, optional): table of benchmark returns. Defaults to None.
        """
        self.returns = returns
        self.benchmark = benchmark

    def summary(self, ann_factor: int = 12, **kwargs) -> pd.DataFrame:
        """Create table with summary statistics.

        Args:
            ann_factor (int): annualisation factor or business days in year. Defaults to 12.

        Returns:
            pd.DataFrame: table with post-trade statistics
        """
        n_obs = self.returns.shape[0]
        t_years = (self.returns.index[-1] - self.returns.index[0]).days / 365.25
        cum_return = self.returns.add(1).cumprod()
        annual_return = cum_return.iloc[-1, :] ** (1 / t_years) - 1
        skew = self.returns.skew()
        kurt = self.returns.kurt() + 3
        volatility = self.returns.std()
        sharpe = self.returns.mean() / volatility  * np.sqrt(ann_factor)
        # se_sr = np.sqrt((1 + np.square(sharpe) / 4 * (kurt - 1) - sharpe * skew) / t_years)
        se_sr = np.sqrt(ann_factor / n_obs * (1 + (kurt - 1) / (4 * ann_factor) * sharpe ** 2))
        maxdd = cum_return.div(cum_return.cummax()).sub(1).min()
        downside_vol = self.returns[self.returns < 0].std()

        summary = OrderedDict(
            {
                "#obs": int(n_obs),
                "#years": t_years,
                "Total Return": cum_return.iloc[-1, :] - 1,
                "Annual. Return": annual_return,
                "Volatility": volatility * np.sqrt(ann_factor),
                "MaxDD": maxdd,
                "Skewness": skew,
                "Kurtosis": kurt,
                "Sharpe Ratio": sharpe,
                "StdErr": se_sr,
                "Sortino": self.returns.mean() / downside_vol * np.sqrt(ann_factor),
                "Calmar": annual_return / maxdd,
                "HWM": cum_return.max(),
                "AutoCorr": autocorrelation(self.returns),
            }
        )
        
        if self.benchmark is not None:
            excess_return = self.returns.drop(self.benchmark, axis=1) - self.returns[self.benchmark]
            summary["InfoRatio"] = np.mean(excess_return) / np.std(excess_return) * ann_factor
        return pd.DataFrame(summary).T
