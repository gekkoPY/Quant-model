# Quant-model

# Quantitative Multi-Asset Portfolio Architecture

A Python-based, institutional-grade algorithmic trading research pipeline. This architecture is designed to evaluate pure macro-trend following strategies across multiple asset classes (Fiat, Commodities) by mathematically neutralizing negative skew and strictly routing capital based on liquidity sessions and fractal market regimes.

## 🧠 Core Quantitative Concepts

* **Fractal Market Regimes:** Uses an OLS-derived **Hurst Exponent** to mathematically classify market environments. Capital is strictly deployed only when the market is in a structural trending regime ($H > 0.55$).
* **Time-Series Momentum (TSMOM):** Replaces moving averages with pure momentum normalized by a 48-period rolling volatility metric.
* **Asymmetric Risk Management:** Utilizes a dynamic, volatility-adjusted trailing stop (Chandelier-style) that ratchets upwards to lock in gains but never moves downwards.
* **The Volatility Floor:** Implements a hard mathematical friction floor to prevent the Volatility Paradox, ensuring dynamic stops never shrink below the broker's spread during dead market hours.
* **Orthogonal Session Routing:** Prevents intraday whipsaw by restricting trade entries exclusively to high-liquidity overlaps (London/NY session), avoiding the mean-reverting chop of the Asian session.
* **Multi-Asset Indexing:** Dynamically aligns asynchronous tick/hourly data from different asset classes (e.g., EURUSD, USDJPY, XAUUSD) into a single, time-synchronized Master Portfolio array to calculate true risk-adjusted metrics.

## ⚙️ Tech Stack & Performance

* **Language:** Python 3.9+
* **Core Libraries:** `pandas`, `numpy`, `plotly`
* **Performance:** Vectorized mathematical operations compiled at the C-level using **Numba (`@jit`)** for near-instantaneous backtesting across decades of hourly data.

## 📁 Repository Structure

```text
├── portfolio_pipeline.py    # The main multi-asset execution engine
├── data_cleaner.py          # (Optional) Utility to format raw MT5 exports
├── EURUSD_Cleaned.csv       # Formatted historical data (User provided)
├── USDJPY_Cleaned.csv       # Formatted historical data (User provided)
├── XAUUSD_Cleaned.csv       # Formatted historical data (User provided)
└── README.md



HOW TO SET IT UP

git clone [https://github.com/Gekko.py/Quantmodel.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME

pip install numpy pandas plotly numba


REMEMBER

This pipeline does not connect to a live API for backtesting to preserve execution speed and environmental sterility. You must provide your own historical data.

Export 1-Hour historical data from MetaTrader 5 (or your preferred broker).

Format the CSV to contain exactly two columns: datetime (e.g., 2023-01-01 14:00:00) and close.

Save the files in the root directory as EURUSD_Cleaned.csv, USDJPY_Cleaned.csv, and XAUUSD_Cleaned.csv.

USAGE

python portfolio_pipeline.py

Expected Output:
The engine will compile the Numba functions, process each asset individually, calculate the isolated performance metrics, and finally merge the time-series arrays to output the Master Fund Performance. A Plotly tearsheet will launch in your browser displaying the equity curve, trade executions, and regime states.

A ledger of all executed trades will be automatically dumped to master_trade_ledger.csv for manual auditing.


Disclaimer
Educational and Research Purposes Only. This repository contains mathematical research architectures. It is not financial advice. The inclusion of broker friction (spreads/commissions) in the backtest highlights the difficulty of retail algorithmic trading. Do not deploy this logic to a live brokerage account without implementing proper Volatility Targeting (Risk Parity sizing) and fully understanding the implications of overnight swap fees, slippage, and tail-risk drawdowns.


