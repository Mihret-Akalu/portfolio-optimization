
```markdown
# Portfolio Optimization using Time Series Forecasting

## Overview

This project implements a complete financial forecasting and portfolio optimization pipeline using:

- Time Series Forecasting (ARIMA, LSTM)
- Modern Portfolio Theory (Efficient Frontier)
- Portfolio Optimization
- Strategy Backtesting

The system forecasts Tesla (TSLA) prices and uses forecasts combined with historical data of SPY and BND to construct an optimal portfolio.

---

## Business Objective

Guide Me in Finance (GMF) Investments wants to:

- Forecast asset trends
- Optimize asset allocation
- Improve returns while managing risk

This project demonstrates how forecasting models can improve portfolio construction.

---

## Assets Used

| Asset | Ticker | Role |
|------|--------|------|
| Tesla | TSLA | High risk / high return |
| Vanguard Bond ETF | BND | Stability |
| S&P500 ETF | SPY | Diversification |

---

## Project Architecture

```

portfolio-optimization/
│
├── data/
│   └── processed/
│
├── models/
│
├── scripts/
│   ├── run_data_pipeline.py
│   ├── run_train_models.py
│   ├── run_forecasting.py
│   ├── run_optimize_portfolio.py
│   ├── run_backtest.py
│
├── src/
│
├── reports/
│   └── figures/
│
└── README.md

````

---

## Pipeline Workflow

### Step 1: Data Pipeline

Fetch and process financial data.

```bash
python -m scripts.run_data_pipeline
````

Output:

```
data/processed/historical_prices.csv
```

---

### Step 2: Train Models

Train:

* ARIMA model
* LSTM model

```bash
python -m scripts.run_train_models
```

Output:

```
models/
tsla_lstm_model.h5
```

---

### Step 3: Forecast Future Prices

```bash
python -m scripts.run_forecasting
```

Output:

```
data/processed/tsla_arima_forecast_with_intervals.csv
```

---

### Step 4: Optimize Portfolio

```bash
python -m scripts.run_optimize_portfolio
```

Example output:

```
TSLA: 49.1%
BND: 30.1%
SPY: 20.8%
```

---

### Step 5: Backtest Strategy

```bash
python -m scripts.run_backtest
```

---

## Results

Example optimized portfolio:

| Asset | Weight |
| ----- | ------ |
| TSLA  | 49.1%  |
| BND   | 30.1%  |
| SPY   | 20.8%  |

---

## Technologies Used

* Python
* pandas
* numpy
* matplotlib
* statsmodels
* tensorflow / keras
* PyPortfolioOpt

---

## Key Features

✔ End-to-end pipeline
✔ Forecasting using ARIMA and LSTM
✔ Efficient Frontier optimization
✔ Portfolio backtesting
✔ Production-ready structure

---

## How to Run Entire Pipeline

```bash
python -m scripts.run_data_pipeline
python -m scripts.run_train_models
python -m scripts.run_forecasting
python -m scripts.run_optimize_portfolio
python -m scripts.run_backtest
```

---

## Author

Mihret Akalu

```
