"""Utility functions such as volatility estimators and statistical measures."""
import logging
from typing import Union

import numpy as np
import pandas as pd
import scipy

from pytaa.tools.logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


def autocorrelation(returns: pd.DataFrame, order: int = 1) -> np.array:
    """Calculate autocorrelation for array of returns.

    Args:
        returns (pd.DataFrame): table of returns
        order (int, optional): time series lag. Defaults to 1.

    Returns:
        np.array: array of autocorrelation coefficients
    """
    merged = returns.merge(returns.shift(order), on="Date", suffixes=["", "_lag"])
    n_assets = returns.shape[1]
    corr = merged.corr().iloc[:n_assets, n_assets:]
    return np.diag(corr)


def calculate_rolling_volatility(
    returns: pd.DataFrame,
    lookback: int = 21,
    factor: int = 252,
    estimator: str = "hist",
    decay: float = 0.97,
) -> pd.DataFrame:
    r"""Calculate historical volatility for all calendar days.

    Currently supported estimators are ``hist`` (realised standard deviation over lookback) and
    ``ewma`` (exponentially weighted with decay factor equal to :math:`1 - \lambda`).

    Args:
        returns (pd.DataFrame): table of asset returns
        lookback (int, optional): window over which vol is calculated. Defaults to 21.
        factor (int, optional): number of business days in a year. Defaults to 252.
        decay (float, optional): half life, or lambda, if using EWMA. Defaults to 0.97.
        estimator (str, optional): volatility estimator. Defaults to hist.

    Returns:
        pd.DataFrame: time series of volatilities matching rebalance dates

    Raises:
        NotImplementedError: when estimator is invalid
    """
    if estimator == "hist":
        rolling_volatility = returns.rolling(lookback).std()
    elif estimator == "ewma":
        rolling_volatility = returns.ewm(alpha=1 - decay).std()
    else:
        raise NotImplementedError
    return rolling_volatility.asfreq("D").ffill() * np.sqrt(factor)


def ledoit_wolf_constant_correlation(
    data: Union[pd.DataFrame, np.array], shrink: float = None
) -> np.array:
    """Shrink sample covariance matrix using Ledoit-Wolf (2003).

    The Matlab code provided by Ledoit and Wolf can be found here:
    https://github.com/oledoit/covShrinkage/blob/main/covCor.m

    Note tha the shrinkage value ``shrink`` will be estimated from the data if it is not provided.

    Args:
        data (Union[pd.DataFrame, np.array]): a table of data
        shrink (float, optional): shrinkage value. Defaults to None.

    Returns:
        np.array: a shrunk covariance matrix
    """
    X = np.asarray(data)
    x = X - X.mean(axis=0)
    t, n = np.shape(X)

    cov_sample = np.cov(X, rowvar=False)

    var = np.diag(cov_sample).reshape(-1, 1)
    std = np.sqrt(var)
    _var = np.tile(var, (n,))
    _std = np.tile(std, (n,))
    r_bar = (np.sum(cov_sample / (_std * _std.T)) - n) / (n * (n - 1))
    prior = r_bar * (_std * _std.T)
    prior[np.diag_indices_from(prior)] = var[:, 0]

    if shrink is None:
        # pi hat
        y = np.square(x)
        pi_mat = y.T @ y / t - 2 * x.T @ x * cov_sample / t + np.square(cov_sample)
        pi_hat = np.sum(pi_mat)

        # rho hat
        term1 = np.power(x, 3).T @ x / t
        help_mat = x.T @ x / t
        help_diag = np.diag(help_mat)
        term2 = np.tile(help_diag, (n, 1)).T * cov_sample
        term3 = help_mat * _var
        term4 = _var * cov_sample
        theta_mat = term1 - term2 - term3 + term4
        theta_mat[np.diag_indices_from(theta_mat)] = np.zeros(n)
        rho_hat = sum(np.diag(pi_mat)) + r_bar * np.sum(((1 / std) @ std.T) * theta_mat)

        # gamma hat
        gamma_hat = np.square(np.linalg.norm(cov_sample - prior, "fro"))

        # calculate shrinkage if not provided
        kappa_hat = (pi_hat - rho_hat) / gamma_hat
        delta = max(0, min(1, kappa_hat / t))
    else:
        delta = shrink

    # calculate final shrunk covariance matrix
    shrunk_cov = delta * prior + (1 - delta) * cov_sample
    return shrunk_cov


class Covariance(np.ndarray):
    def __new__(
        cls,
        data: pd.DataFrame,
        shrinkage: str = None,
        shrinkage_factor: float = None,
        **kwargs: dict
    ):
        """Initialise covariance matrix.

        Currently the following shrinkage factors are supported: ``None``, ``ledoit_wolf``,
        ``constant`` and ``weighted``. Constant shrinkage shrinks the covar matrix towards the
        identity matrix. If using constant shrinkage, then a shrinkage factor must be provided.
        The weighted method follows Ilya Kipnis (2019).

        Args:
            data (pd.DataFrame): data table
            shrinkage (str, optional): shrinkage method. Defaults to None.
            shrinkage_factor (float, optional): shrinkage factor between 0 and 1. Defaults to None.
            **kwargs (dict): additional keyword arguments for estimation method
        """
        self = np.asarray(data).view(cls)
        self.shrinkage = shrinkage
        self.shrinkage_factor = shrinkage_factor
        return self._estimate(**kwargs)

    def _estimate(self, **kwargs) -> np.array:
        """Estimate variance-covariance matrix.

        Raises:
            NotImplementedError: if shrinkage method is invalid

        Returns:
            np.array: NxN covariance matrix
        """

        if self.shrinkage is None:
            return np.cov(self, rowvar=False)
        elif self.shrinkage == "ledoit_wolf":
            return ledoit_wolf_constant_correlation(self, self.shrinkage_factor)
        elif self.shrinkage == "constant":
            sample_cov = np.cov(self, rowvar=False)
            target_cov = (1 - self.shrinkage_factor) * np.eye(sample_cov.shape[1])
            return self.shrinkage_factor * sample_cov + target_cov
        elif self.shrinkage == "weighted":
            return weighted_covariance_matrix(self, **kwargs)
        else:
            raise NotImplementedError


def calculate_risk_parity_portfolio(
    cov: Union[Covariance, np.array], risk_budget: np.array = None
) -> np.array:
    """Allocate risk equally without leverage.

    The methodology can be found in  Maillard, Roncalli & Teiletche (2009):
    http://www.thierry-roncalli.com/download/erc.pdf

    Args:
        cov (Union[Covariance, np.array]): covariance matrix
        risk_budget(np.array): vector of risk bugets summing up to 1

    Returns:
        np.array: a 1d vector of weights
    """
    inv_sigma = 1 / np.sqrt(np.diag(cov))
    initial_weights = inv_sigma / np.sum(inv_sigma)

    if risk_budget is None:
        risk_budget = np.ones((cov.shape[1], 1)) / cov.shape[1]

    c_max = -cov.shape[0] * np.log(cov.shape[0])

    constraints = (
        {"type": "ineq", "fun": lambda w: w},
        {"type": "ineq", "fun": lambda w: np.sum(risk_budget * np.log(w)) - c_max},
    )

    result = scipy.optimize.minimize(
        lambda w, S: np.sqrt(w.T @ S @ w),
        x0=initial_weights,
        args=(cov),
        method="SLSQP",
        constraints=constraints,
        options={"disp": False},
        tol=1e-9,
    )
    optimal_weights = result.x.flatten() / np.sum(result.x.flatten())
    if result.success:
        return optimal_weights
    logger.warn("Risk Parity optimization failed: returning inverse vol weights.")
    return initial_weights


def calculate_min_variance_portfolio(cov: Union[Covariance, np.array]) -> np.array:
    """Calculate minimum variance portfolio with no leverage.

    Args:
        cov (Union[Covariance, np.array]): covariance matrix

    Returns:
        np.array: a 1d vector of weights
    """
    inv_sigma = 1 / np.sqrt(np.diag(cov))
    initial_weights = inv_sigma / np.sum(inv_sigma)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)

    result = scipy.optimize.minimize(
        lambda w, S: w.T @ S @ w,
        initial_weights,
        args=[cov],
        method="SLSQP",
        bounds=scipy.optimize.Bounds(0, 1),
        constraints=constraints,
        options={"disp": False},
        tol=1e-9,
    )

    if result.success:
        return result.x.flatten()
    logger.warn("Min variance optimization failed: returning inverse vol weights.")
    return initial_weights


def calculate_mean_variance_portfolio(mu: np.array, cov: Union[Covariance, np.array]) -> np.array:
    """Calculate mean-variance portfolio with no leverage.

    Args:
        mu (np.array): vector of expected returns
        cov (Union[Covariance, np.array]): covariance matrix

    Returns:
        np.array: a 1d vector of weights
    """
    initial_weights = np.ones(cov.shape[0]) / cov.shape[0]

    constraints = (
        {"type": "ineq", "fun": lambda w: w},
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
    )

    result = scipy.optimize.minimize(
        lambda w, S, mu: w.T @ S @ w - w.T @ mu,
        initial_weights,
        args=[cov, mu],
        method="SLSQP",
        bounds=scipy.optimize.Bounds(0, 1),
        constraints=constraints,
        options={"disp": False},
        tol=1e-9,
    )

    if result.success:
        return result.x.flatten()
    logger.warn("Mean variance optimization failed: returning equal weights.")
    return initial_weights


def risk_contribution(weights: np.array, cov: np.array) -> np.array:
    """Calculate ex-post risk contribution of portfolio allocation.

    Args:
        weights (np.array): Mx1 vector of weight
        cov (np.array): MxM covariance matrix

    Returns:
        np.array: Mx1 vector of portfolio risk contributions
    """
    port_vol = np.sqrt(weights.T @ cov @ weights)
    marginal_risk_contrib = cov @ weights
    port_risk_contrib = marginal_risk_contrib * weights / port_vol
    return port_risk_contrib


# TODO: make this a class WeightedCovariance similar to above
def weighted_covariance_matrix(
    data: pd.DataFrame, weights: list = (12, 4, 2, 1), vol_window: int = 21, ann_factor: int = 252
) -> np.array:
    """Calculate weighted covariance matrix.
    
    This technique is used in Kipnis Asset Allocation. Each weight in the ``weights`` list refers
    to a number of months. The weighted correlation matrix is first calculated as:
    
    R = [(21-day correlation * 12) + (63-day correlation * 4) + (126-day correlation * 2) + \
        (252-day correlation)] / 19

    Then the covariance matrix is calculated using 21-day volatility: diag(vol) .* R .* diag(vol).

    More details can be found on:
    https://quantstrattrader.com/2019/01/24/right-now-its-kda-asset-allocation/

    Args:
        data (pd.DataFrame): table of returns
        weights (list): list of weights in months. Defaults to [12, 4, 2, 1].
        vol_window (int, optional): sample size of volatility. Defaults to 21.
        ann_factor (int, optional): number of days in year. Defaults to 252.

    Returns:
        np.array: variance covariance matrix
    """
    weighted_corr = None
    _data = np.asarray(data)

    for window in weights:
        lookback = int(ann_factor / window)
        sample_corr = window * np.corrcoef(_data[-lookback:, :], rowvar=False)

        if weighted_corr is None:
            weighted_corr = sample_corr
        else:
            weighted_corr += sample_corr

    weighted_corr = sample_corr / np.sum(weights)
    sample_vol = np.std(_data[-vol_window:, :], axis=0)
    return np.diag(sample_vol) @ weighted_corr @ np.diag(sample_vol)
