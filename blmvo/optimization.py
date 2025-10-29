"""Mean-variance optimisation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cvxpy as cp
import numpy as np


@dataclass
class FrontierResult:
    returns: np.ndarray
    risks: np.ndarray
    weights: np.ndarray


def mean_variance_weights(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    risk_aversion: float | None = None,
    target_return: float | None = None,
    allow_short: bool = False,
) -> np.ndarray:
    """Solve for optimal portfolio weights using mean-variance optimisation."""

    mu = np.asarray(expected_returns, dtype=float)
    Sigma = np.asarray(covariance, dtype=float)
    n = mu.shape[0]
    w = cp.Variable(n)

    objective = None
    constraints = [cp.sum(w) == 1]
    if not allow_short:
        constraints.append(w >= 0)

    if target_return is not None:
        constraints.append(mu @ w >= target_return)
        objective = cp.Minimize(cp.quad_form(w, Sigma))
    else:
        if risk_aversion is None:
            risk_aversion = 1.0
        objective = cp.Maximize(mu @ w - risk_aversion * cp.quad_form(w, Sigma))

    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.OSQP, verbose=False)
    except cp.SolverError:
        problem.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        raise ValueError("Optimisation failed to find a solution.")
    weights = np.asarray(w.value).flatten()
    total = weights.sum()
    if total != 0:
        weights = weights / total
    return weights


def efficient_frontier(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    allow_short: bool = False,
    points: int = 25,
) -> FrontierResult:
    """Compute the efficient frontier by varying the target return."""

    mu = np.asarray(expected_returns, dtype=float)
    Sigma = np.asarray(covariance, dtype=float)

    min_return = np.min(mu)
    max_return = np.max(mu)
    targets = np.linspace(min_return, max_return, points)

    weights = []
    risks = []
    realised_returns = []
    for target in targets:
        try:
            w = mean_variance_weights(
                expected_returns=mu,
                covariance=Sigma,
                target_return=target,
                allow_short=allow_short,
            )
        except ValueError:
            continue
        variance = float(w.T @ Sigma @ w)
        risks.append(np.sqrt(variance))
        realised_returns.append(float(w @ mu))
        weights.append(w)

    return FrontierResult(
        returns=np.array(realised_returns),
        risks=np.array(risks),
        weights=np.array(weights),
    )


def portfolio_performance(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    risk_free_rate: float = 0.0,
) -> Tuple[float, float, float]:
    """Compute expected return, volatility and Sharpe ratio."""

    w = np.asarray(weights, dtype=float)
    mu = np.asarray(expected_returns, dtype=float)
    Sigma = np.asarray(covariance, dtype=float)

    expected = float(w @ mu)
    variance = float(w.T @ Sigma @ w)
    volatility = float(np.sqrt(variance))
    sharpe = (expected - risk_free_rate) / volatility if volatility > 0 else np.nan

    return expected, volatility, sharpe
