"""Utility functions such as volatility estimators and statistical measures."""
from typing import Union

import numpy as np
import pandas as pd
import scipy


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
    """Calculate historical volatility for all calendar days.

    Currently supported estimators are ``hist`` (realised standard deviation over lookback) and
    ``ewm`` (exponentially weighted with decay factor equal to :math:`1 - \lambda`).

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
    elif estimator == "ewm":
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


def calculate_risk_parity(
    data: np.array, shrinkage_method: str = None, shrinkage_factor: float = None
) -> np.array:
    """Allocate risk equally without leverage.

    Shrinkage can be one of [``ledoit_wolf``, ``constant``]. Constant shrinkage shrinks the covar
    matrix towards the identity matrix. If using constant shrinkage, then a shrinkage factor must
    be provided.

    Args:
        data (np.array): table of returns
        shrinkage_method (str, optional): shrinkage method. Defaults to None.
        shrinkage_factor (str, optional): shrinkage factor. Defaults to None.

    Returns:
        np.array: a 1d vector of weights
    """
    if shrinkage_method is None:
        cov = np.cov(data, rowvar=False)
    elif shrinkage_method == "ledoit_wolf":
        cov = ledoit_wolf_constant_correlation(data, shrinkage_factor)
    elif shrinkage_factor == "constant":
        sample_cov = np.cov(data, rowvar=False)
        cov = shrinkage_factor * sample_cov + (1 - shrinkage_factor) * np.eye(sample_cov.shape[1])
    else:
        raise NotImplementedError

    initial_weights = 1 / np.sqrt(np.diag(cov).reshape(-1, 1))

    result = scipy.optimize.minimize(
        lambda w, S: np.sqrt(w.T @ S @ w) - np.sum(np.log(w)),
        x0=1 / np.ones(cov.shape[0]),
        args=[cov],
        method="SLSQP",
        bounds=scipy.optimize.Bounds(1),
        constraints=({"type": "eq", "fun": lambda w: np.sum(w) - 1}),
        options={"disp": False},
    )

    if result.success:
        return result.x.flatten()
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
