
# Portfolio Optimization and Forecasting – TSLA, BND, SPY

## Project Overview
This project demonstrates a data-driven approach to **financial market analysis**, **forecasting**, and **portfolio optimization**. Using historical stock and bond data, we generate forecasts for Tesla (TSLA), optimize a portfolio including TSLA, BND, and SPY based on Modern Portfolio Theory, and backtest the strategy against a benchmark portfolio.

The project is divided into five main tasks:

1. **Data Preprocessing & EDA** – Cleaning and exploratory analysis of historical market data.
2. **Forecasting Models** – Building and evaluating models (ARIMA/LSTM) to predict TSLA stock prices.
3. **Future Market Trends Forecasting** – Generating 6–12 month forecasts with confidence intervals and trend analysis.
4. **Portfolio Optimization** – Constructing an optimal portfolio using Modern Portfolio Theory.
5. **Strategy Backtesting** – Simulating portfolio performance and comparing against a benchmark.

---

## Project Structure

```

portfolio-optimization/
│
├─ notebooks/
│   ├─ 01_data_preprocessing_eda.ipynb        # Task 1: Data cleaning & EDA
│   ├─ 02_forecasting_models.ipynb            # Task 2: ARIMA/LSTM models
│   ├─ 03_forecast_future.ipynb               # Task 3: Future forecast & trend analysis
│   ├─ 04_portfolio_optimization.ipynb        # Task 4: Portfolio optimization
│   ├─ 05_backtesting.ipynb                   # Task 5: Backtesting
│
├─ data/
│   ├─ raw/                                   # Raw downloaded CSVs
│   ├─ processed/                             # Cleaned and processed datasets
│
├─ venv/                                      # Python virtual environment
├─ .gitignore                                 # Git ignore file
├─ README.md                                  # Project description (this file)

````

---

## Setup Instructions

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd portfolio-optimization
````

2. **Create and activate a virtual environment:**

```bash
python -m venv venv

venv\Scripts\activate

```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Run Jupyter notebooks:**

```bash
jupyter notebook
```

---

## Key Libraries

* `pandas`, `numpy` – Data manipulation and numerical computation
* `matplotlib`, `seaborn` – Data visualization
* `statsmodels`, `pmdarima` – Time series modeling (ARIMA)
* `tensorflow/keras` – Deep learning (LSTM models)
* `PyPortfolioOpt` – Portfolio optimization (Efficient Frontier, Sharpe ratio)
* `scikit-learn` – Forecast evaluation metrics
* `yfinance` – Historical financial data download

---

## Tasks Summary

### **Task 1: Data Preprocessing & EDA**

* Cleaned TSLA, BND, SPY historical data
* Handled missing values and aligned dates
* Visualized price trends, daily returns, and correlations
* Stationarity testing using ADF Test

### **Task 2: Forecasting Models**

* Trained ARIMA and/or LSTM models for TSLA price prediction
* Evaluated models using MAE, RMSE, and MAPE
* Visualized forecast vs. actual

### **Task 3: Future Market Trends Forecasting**

* Generated 6–12 month forecasts using the best model
* Plotted forecasts with 95% confidence intervals
* Performed trend analysis and assessed market opportunities and risks

### **Task 4: Portfolio Optimization**

* Constructed expected returns vector (forecasted TSLA + historical BND/SPY)
* Calculated covariance matrix and risk metrics
* Generated efficient frontier
* Identified Max Sharpe Ratio and Min Volatility portfolios
* Recommended optimal portfolio weights

### **Task 5: Strategy Backtesting**

* Backtested strategy using last year of historical data
* Compared cumulative returns against benchmark portfolio (60% SPY / 40% BND)
* Calculated key performance metrics (total return, Sharpe ratio, max drawdown)
* Provided insights on strategy viability and limitations

---

## Usage

1. Start with **Task 1 notebook** to preprocess and visualize data.
2. Run **Task 2 notebook** to train forecasting models.
3. Use **Task 3 notebook** for future predictions and trend analysis.
4. Optimize the portfolio with **Task 4 notebook**.
5. Backtest the strategy using **Task 5 notebook**.

---

## Notes & Recommendations

* Always ensure `data/processed/` contains up-to-date cleaned datasets before running forecasting or portfolio optimization.
* Use consistent date alignment across all assets for accurate covariance and returns calculation.
* Forecast reliability decreases over longer horizons; always check confidence intervals.
* The backtest is **historical simulation only**; past performance does not guarantee future results.



