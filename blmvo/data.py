"""Utility helpers for loading and preparing data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_price_data(path: str | Path) -> pd.DataFrame:
    """Load price history from a CSV file."""

    path = Path(path)
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    df.index.name = "Date"
    return df


def compute_sample_statistics(prices: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Return sample mean returns and covariance matrix from prices."""

    returns = prices.pct_change().dropna()
    mu = returns.mean() * 12
    cov = returns.cov() * 12
    return mu, cov
