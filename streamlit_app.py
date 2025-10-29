"""Interactive Streamlit dashboard for Black-Litterman optimisation."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from blmvo.backtest import backtest_static_portfolio
from blmvo.black_litterman import View, black_litterman_posterior, build_views_matrices
from blmvo.data import compute_sample_statistics, load_price_data
from blmvo.optimization import (
    efficient_frontier,
    mean_variance_weights,
    portfolio_performance,
)

DATA_PATH = Path("blmvo/data/sample_prices.csv")


def _normalise_weights(weights: pd.Series) -> pd.Series:
    total = weights.sum()
    if total == 0:
        return pd.Series(np.repeat(1 / len(weights), len(weights)), index=weights.index)
    return weights / total


def _views_from_editor(df: pd.DataFrame) -> List[View]:
    views: List[View] = []
    for _, row in df.iterrows():
        view_type = row.get("type")
        if pd.isna(view_type):
            continue
        try:
            value = float(row.get("value", 0.0))
            confidence = float(row.get("confidence", 50.0))
        except (TypeError, ValueError):
            continue
        view = View(
            type=str(view_type).lower(),
            value=value,
            confidence=confidence,
            asset=row.get("asset"),
            asset_long=row.get("asset_long"),
            asset_short=row.get("asset_short"),
        )
        views.append(view)
    return views


def _plot_efficient_frontier(frontier, label: str, colour: str) -> go.Scatter:
    return go.Scatter(
        x=frontier.risks,
        y=frontier.returns,
        mode="lines+markers",
        name=label,
        line=dict(color=colour),
    )


def main() -> None:
    st.set_page_config(page_title="Black-Litterman Optimiser", layout="wide")
    st.title("Black-Litterman Portfolio Optimisation")

    st.sidebar.header("Data & Parameters")
    uploaded = st.sidebar.file_uploader("Upload price CSV", type=["csv"])
    if uploaded is not None:
        prices = pd.read_csv(uploaded, index_col=0, parse_dates=True)
    else:
        prices = load_price_data(DATA_PATH)

    st.sidebar.caption("Using monthly price data. Returns annualised for optimisation.")

    mu_sample, cov_sample = compute_sample_statistics(prices)
    assets = list(mu_sample.index)

    delta = st.sidebar.slider("Risk aversion (δ)", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
    tau = st.sidebar.slider("Tau (scales prior uncertainty)", min_value=0.01, max_value=0.5, value=0.05, step=0.01)

    st.sidebar.subheader("Market Weights")
    default_weights = pd.Series(np.repeat(1 / len(assets), len(assets)), index=assets)
    weight_inputs = {}
    for asset in assets:
        weight_inputs[asset] = st.sidebar.number_input(
            f"{asset}", min_value=0.0, max_value=1.0, value=float(default_weights[asset]), step=0.01
        )
    market_weights = _normalise_weights(pd.Series(weight_inputs))

    st.sidebar.subheader("Backtest baseline")
    baseline_choice = st.sidebar.selectbox("Baseline portfolio", ["Market Weights", "Mean-Variance"], index=0)

    st.subheader("Investor Views")
    st.markdown(
        "Specify absolute or relative views on the assets.\n"
        "Absolute view expects an annualised return for a single asset.\n"
        "Relative view expresses the expected outperformance of one asset over another."
    )

    view_editor = st.data_editor(
        pd.DataFrame(
            {
                "type": [],
                "asset": [],
                "asset_long": [],
                "asset_short": [],
                "value": [],
                "confidence": [],
            }
        ),
        num_rows="dynamic",
        column_config={
            "type": st.column_config.SelectboxColumn("View Type", options=["absolute", "relative"]),
            "asset": st.column_config.SelectboxColumn("Asset", options=assets),
            "asset_long": st.column_config.SelectboxColumn("Long Asset", options=assets),
            "asset_short": st.column_config.SelectboxColumn("Short Asset", options=assets),
            "value": st.column_config.NumberColumn("Value", format="%.4f"),
            "confidence": st.column_config.NumberColumn("Confidence (0-100)", min_value=0, max_value=100),
        },
        use_container_width=True,
    )

    views = _views_from_editor(view_editor)

    if views:
        try:
            P, Q, Omega = build_views_matrices(assets, views, cov_sample.values, tau)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Error constructing views: {exc}")
            P = Q = Omega = None
    else:
        P = Q = Omega = None

    pi, posterior_mean, posterior_cov = black_litterman_posterior(
        covariance=cov_sample.values,
        market_weights=market_weights.values,
        risk_aversion=delta,
        tau=tau,
        P=P,
        Q=Q,
        Omega=Omega,
    )

    st.subheader("Equilibrium & Posterior Returns")
    returns_df = pd.DataFrame(
        {
            "Sample Mean": mu_sample,
            "Implied (π)": pd.Series(pi, index=assets),
            "Posterior": pd.Series(posterior_mean, index=assets),
        }
    )
    st.dataframe(returns_df.style.format("{:.4%}"))

    st.subheader("Optimised Portfolio Weights")
    # The mean-variance objective implemented in ``mean_variance_weights`` uses
    # ``mu @ w - penalty * w'Σw``. For a classical quadratic utility of
    # ``mu @ w - δ/2 * w'Σw`` we therefore supply ``δ / 2`` as the penalty.
    mv_penalty = delta / 2.0

    mv_weights = mean_variance_weights(mu_sample.values, cov_sample.values, risk_aversion=mv_penalty)
    bl_weights = mean_variance_weights(posterior_mean, posterior_cov, risk_aversion=mv_penalty)

    weights_df = pd.DataFrame(
        {
            "Market": market_weights,
            "Mean-Variance": mv_weights,
            "Black-Litterman": bl_weights,
        },
        index=assets,
    )
    st.dataframe(weights_df.style.format("{:.2%}"))

    mv_perf = portfolio_performance(mv_weights, mu_sample.values, cov_sample.values)
    bl_perf = portfolio_performance(bl_weights, posterior_mean, posterior_cov)

    active_diff = bl_weights - market_weights.values
    active_risk = float(np.sqrt(active_diff.T @ cov_sample.values @ active_diff))

    st.markdown("### Risk & Return Summary")
    metrics_df = pd.DataFrame(
        {
            "Expected Return": [mv_perf[0], bl_perf[0]],
            "Volatility": [mv_perf[1], bl_perf[1]],
            "Sharpe": [mv_perf[2], bl_perf[2]],
        },
        index=["Mean-Variance", "Black-Litterman"],
    )
    st.dataframe(
        metrics_df.style.format({"Expected Return": "{:.2%}", "Volatility": "{:.2%}", "Sharpe": "{:.2f}"})
    )
    st.metric("Active Risk vs Market", f"{active_risk:.2%}")

    st.markdown("### Efficient Frontier Comparison")
    mv_frontier = efficient_frontier(mu_sample.values, cov_sample.values)
    bl_frontier = efficient_frontier(posterior_mean, posterior_cov)
    fig = go.Figure()
    fig.add_trace(_plot_efficient_frontier(mv_frontier, "Mean-Variance", "#1f77b4"))
    fig.add_trace(_plot_efficient_frontier(bl_frontier, "Black-Litterman", "#ff7f0e"))
    fig.update_layout(
        xaxis_title="Volatility",
        yaxis_title="Expected Return",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Backtest")
    if baseline_choice == "Market Weights":
        baseline_weights = market_weights.values
    else:
        baseline_weights = mv_weights
    backtest = backtest_static_portfolio(prices, bl_weights, baseline_weights)

    st.line_chart(backtest.cumulative_returns)
    st.dataframe(
        backtest.summary.style.format(
            {
                "Annualised Return": "{:.2%}",
                "Annualised Volatility": "{:.2%}",
                "Sharpe (rf=0%)": "{:.2f}",
            }
        )
    )

    st.sidebar.download_button(
        "Download posterior returns",
        data=returns_df.to_csv(float_format="%.6f").encode("utf-8"),
        file_name="posterior_returns.csv",
        mime="text/csv",
    )

    st.sidebar.download_button(
        "Download weights",
        data=weights_df.to_csv(float_format="%.6f").encode("utf-8"),
        file_name="portfolio_weights.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
