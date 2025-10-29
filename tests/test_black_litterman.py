"""Unit tests for the Black-Litterman implementation."""

from __future__ import annotations

import numpy as np

from blmvo.black_litterman import (
    View,
    black_litterman_posterior,
    build_views_matrices,
    compute_market_implied_returns,
)


def _example_inputs():
    covariance = np.array(
        [
            [0.04, 0.006, 0.012],
            [0.006, 0.09, 0.018],
            [0.012, 0.018, 0.16],
        ]
    )
    weights = np.array([0.6, 0.3, 0.1])
    risk_aversion = 2.5
    tau = 0.05
    assets = ["A", "B", "C"]
    views = [
        View(type="absolute", value=0.07, confidence=60, asset="A"),
        View(
            type="relative",
            value=0.02,
            confidence=55,
            asset_long="B",
            asset_short="C",
        ),
    ]
    return covariance, weights, risk_aversion, tau, assets, views


def test_market_implied_returns_matches_manual_computation():
    covariance, weights, risk_aversion, _, _, _ = _example_inputs()
    pi = compute_market_implied_returns(covariance, weights, risk_aversion)
    expected = risk_aversion * covariance @ weights
    np.testing.assert_allclose(pi, expected)


def test_black_litterman_no_views_reverts_to_prior():
    covariance, weights, risk_aversion, tau, *_ = _example_inputs()
    pi, posterior_mean, posterior_cov = black_litterman_posterior(
        covariance=covariance,
        market_weights=weights,
        risk_aversion=risk_aversion,
        tau=tau,
    )
    np.testing.assert_allclose(posterior_mean, pi)
    np.testing.assert_allclose(posterior_cov, covariance + tau * covariance)


def test_black_litterman_matches_closed_form_solution():
    covariance, weights, risk_aversion, tau, assets, views = _example_inputs()
    P, Q, Omega = build_views_matrices(assets, views, covariance, tau)

    pi, posterior_mean, posterior_cov = black_litterman_posterior(
        covariance=covariance,
        market_weights=weights,
        risk_aversion=risk_aversion,
        tau=tau,
        P=P,
        Q=Q,
        Omega=Omega,
    )

    tau_sigma = tau * covariance
    gain = tau_sigma @ P.T
    M = P @ gain + Omega
    diff = Q - P @ pi
    manual_mean = pi + gain @ np.linalg.solve(M, diff)
    manual_cov = covariance + tau_sigma - gain @ np.linalg.solve(M, P @ tau_sigma)

    np.testing.assert_allclose(posterior_mean, manual_mean)
    np.testing.assert_allclose(posterior_cov, manual_cov)

