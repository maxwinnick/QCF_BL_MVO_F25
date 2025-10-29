# Black-Litterman Mean-Variance Optimisation Toolkit

This project provides a from-scratch implementation of the Black-Litterman portfolio construction framework together with a teaching-oriented Streamlit dashboard. The toolkit can be used to explore how investor views, confidence levels, and Bayesian updates interact with traditional mean-variance optimisation.

## Features

- Reverse optimisation to compute market equilibrium returns from covariance matrices, market weights and risk-aversion.
- Support for absolute and relative investor views with 0–100% confidence scores translated into view-uncertainty values.
- Full Bayesian Black-Litterman update to produce posterior return estimates and covariance matrices.
- Mean-variance optimisation utilities, efficient frontier generation and risk/return diagnostics.
- Backtesting utilities to benchmark Black-Litterman allocations against market-cap or sample mean-variance portfolios.
- Interactive Streamlit dashboard enabling asset selection, view specification, confidence adjustment and real-time visualisation of outcomes.

## Project structure

```
blmvo/
├── __init__.py
├── backtest.py
├── black_litterman.py
├── data.py
├── optimization.py
└── data/
    └── sample_prices.csv
streamlit_app.py
```

The `blmvo` package holds reusable Python components for data preparation, Black-Litterman algebra, optimisation, and backtesting. `streamlit_app.py` wires everything into an interactive user interface. A small synthetic data set is available under `blmvo/data/sample_prices.csv` for experimentation without external data.

## Getting started

1. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch the Streamlit dashboard:

   ```bash
   streamlit run streamlit_app.py
   ```

   The application loads the bundled sample data by default. You can upload your own price history via CSV, adjust market weights and parameters from the sidebar, and add investor views in the main pane. The dashboard surfaces posterior returns, optimal allocations, efficient frontiers, and a backtest relative to the chosen baseline portfolio.

## Usage as a library

The computational building blocks can be imported directly for scripting or research workflows:

```python
import numpy as np
import pandas as pd

from blmvo.black_litterman import View, build_views_matrices, black_litterman_posterior
from blmvo.data import compute_sample_statistics
from blmvo.optimization import mean_variance_weights

prices = pd.read_csv("blmvo/data/sample_prices.csv", index_col=0, parse_dates=True)
mu, cov = compute_sample_statistics(prices)
market_weights = np.repeat(1 / len(mu), len(mu))
views = [View(type="absolute", asset=mu.index[0], value=0.08, confidence=75)]
P, Q, Omega = build_views_matrices(mu.index, views, cov.values, tau=0.05)
pi, posterior_mu, posterior_cov = black_litterman_posterior(cov.values, market_weights, risk_aversion=3.0, tau=0.05, P=P, Q=Q, Omega=Omega)
weights = mean_variance_weights(posterior_mu, posterior_cov, risk_aversion=1.0)
```

## Requirements

Dependencies are listed in `requirements.txt`. Key libraries include:

- `numpy`, `pandas` for numerical operations.
- `cvxpy` for quadratic optimisation.
- `plotly` and `streamlit` for visualisation and the interactive front-end.

## License

Distributed under the terms of the MIT License. See `LICENSE` for details.
