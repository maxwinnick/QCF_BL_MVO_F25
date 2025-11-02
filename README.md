# Black-Litterman Mean-Variance Optimization Toolkit

This project provides a from-scratch implementation of the Black-Litterman portfolio construction framework together with a teaching-oriented Streamlit dashboard. The toolkit can be used to explore how investor views, confidence levels, and Bayesian updates interact with traditional mean-variance optimisation.

## Features

- Reverse optimisation to compute market equilibrium returns from covariance matrices, market weights and risk-aversion.
- Support for absolute and relative investor views with 0–100% confidence scores translated into view-uncertainty values.
- Full Bayesian Black-Litterman update to produce posterior return estimates and covariance matrices.
- Mean-variance optimisation utilities, efficient frontier generation and risk/return diagnostics.
- Backtesting utilities to benchmark Black-Litterman allocations against market-cap or sample mean-variance portfolios.
- Interactive Streamlit dashboard enabling asset selection, view specification, confidence adjustment and real-time visualisation of outcomes.

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


## Requirements

Dependencies are listed in `requirements.txt`. Key libraries include:

- `numpy`, `pandas` for numerical operations.
- `cvxpy` for quadratic optimisation.
- `plotly` and `streamlit` for visualisation and the interactive front-end.

## License

Distributed under the terms of the MIT License. See `LICENSE` for details.
