"""Core Black-Litterman implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np


@dataclass
class View:
    """Container describing an investor view."""

    type: str
    value: float
    confidence: float
    asset: str | None = None
    asset_long: str | None = None
    asset_short: str | None = None

    def to_loading(self, asset_index: Dict[str, int]) -> np.ndarray:
        """Convert the view into a P-row (factor loading)."""

        n = len(asset_index)
        p = np.zeros(n)
        if self.type == "absolute":
            if not self.asset:
                raise ValueError("Absolute view requires 'asset'.")
            p[asset_index[self.asset]] = 1.0
        elif self.type == "relative":
            if not (self.asset_long and self.asset_short):
                raise ValueError("Relative view requires 'asset_long' and 'asset_short'.")
            p[asset_index[self.asset_long]] = 1.0
            p[asset_index[self.asset_short]] = -1.0
        else:
            raise ValueError(f"Unsupported view type: {self.type}")
        return p


def compute_market_implied_returns(
    covariance: np.ndarray,
    market_weights: np.ndarray,
    risk_aversion: float,
) -> np.ndarray:
    """Reverse optimization to obtain equilibrium (implied) returns."""

    market_weights = np.asarray(market_weights, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Covariance matrix must be square.")
    if covariance.shape[0] != market_weights.shape[0]:
        raise ValueError("Weights and covariance dimension mismatch.")
    return risk_aversion * covariance @ market_weights


def _confidence_to_variance(
    p_row: np.ndarray,
    tau_sigma: np.ndarray,
    confidence: float,
    min_confidence: float = 1e-3,
) -> float:
    """Map a 0-100 confidence score to view uncertainty."""

    c = max(min(confidence, 100.0), 0.0) / 100.0
    base_var = p_row @ tau_sigma @ p_row.T
    if base_var <= 0:
        base_var = min_confidence
    if c <= min_confidence:
        return 1e9
    scale = (1.0 - c) / c
    variance = base_var * scale
    return max(variance, min_confidence)


def build_views_matrices(
    assets: Sequence[str],
    views: Iterable[View],
    covariance: np.ndarray,
    tau: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct the P, Q and Omega matrices from structured views."""

    assets = list(assets)
    asset_index = {asset: i for i, asset in enumerate(assets)}
    views = list(views)
    if not views:
        raise ValueError("At least one view is required to build matrices.")

    n_views = len(views)
    n_assets = len(assets)
    P = np.zeros((n_views, n_assets))
    Q = np.zeros(n_views)
    Omega = np.zeros((n_views, n_views))

    tau_sigma = tau * covariance

    for i, view in enumerate(views):
        p_row = view.to_loading(asset_index)
        P[i, :] = p_row
        Q[i] = view.value
        Omega[i, i] = _confidence_to_variance(p_row, tau_sigma, view.confidence)

    return P, Q, Omega


def black_litterman_posterior(
    covariance: np.ndarray,
    market_weights: np.ndarray,
    risk_aversion: float,
    tau: float,
    P: np.ndarray | None = None,
    Q: np.ndarray | None = None,
    Omega: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the implied returns, posterior returns and covariance."""

    covariance = np.asarray(covariance, dtype=float)
    market_weights = np.asarray(market_weights, dtype=float)
    pi = compute_market_implied_returns(covariance, market_weights, risk_aversion)

    if P is None or Q is None or Omega is None or len(P) == 0:
        posterior_mean = pi
        posterior_cov = covariance
    else:
        tau_sigma = tau * covariance
        inv_tau_sigma = np.linalg.inv(tau_sigma)
        inv_omega = np.linalg.inv(Omega)
        middle_term = np.linalg.inv(inv_tau_sigma + P.T @ inv_omega @ P)
        posterior_mean = middle_term @ (inv_tau_sigma @ pi + P.T @ inv_omega @ Q)
        posterior_cov = covariance + middle_term

    return pi, posterior_mean, posterior_cov
