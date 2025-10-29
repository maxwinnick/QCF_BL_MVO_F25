"""Backtesting utilities for static portfolios."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    cumulative_returns: pd.DataFrame
    summary: pd.DataFrame


def _portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    w = np.asarray(weights, dtype=float)
    return returns @ w


def _annualise_return(series: pd.Series) -> float:
    compounded = (1 + series).prod()
    periods = len(series)
    return compounded ** (12 / periods) - 1


def _annualise_vol(series: pd.Series) -> float:
    return float(series.std() * np.sqrt(12))


def _sharpe(series: pd.Series) -> float:
    vol = _annualise_vol(series)
    return (series.mean() * 12) / vol if vol > 0 else np.nan


def backtest_static_portfolio(
    prices: pd.DataFrame,
    candidate_weights: np.ndarray,
    baseline_weights: np.ndarray,
) -> BacktestResult:
    """Compute cumulative performance for candidate vs baseline portfolios."""

    prices = prices.sort_index()
    returns = prices.pct_change().dropna()
    candidate = _portfolio_returns(returns, candidate_weights)
    baseline = _portfolio_returns(returns, baseline_weights)
    diff = candidate - baseline

    cumulative = pd.DataFrame(
        {
            "Candidate": (1 + candidate).cumprod(),
            "Baseline": (1 + baseline).cumprod(),
            "Excess": (1 + diff).cumprod(),
        },
        index=returns.index,
    )

    summary = pd.DataFrame(
        {
            "Annualised Return": [
                _annualise_return(candidate),
                _annualise_return(baseline),
            ],
            "Annualised Volatility": [
                _annualise_vol(candidate),
                _annualise_vol(baseline),
            ],
            "Sharpe (rf=0%)": [
                _sharpe(candidate),
                _sharpe(baseline),
            ],
        },
        index=["Candidate", "Baseline"],
    )

    return BacktestResult(cumulative_returns=cumulative, summary=summary)
