import pandas as pd
import matplotlib.pyplot as plt
import os

# Load historical data
tsla = pd.read_csv("data/processed/historical_prices.csv")

# Load ARIMA forecast
arima = pd.read_csv("data/processed/tsla_arima_forecast_with_intervals.csv")

# Convert Date column to datetime
tsla["Date"] = pd.to_datetime(tsla["Date"])

# Create forecast index
forecast_index = range(len(tsla), len(tsla) + len(arima))

# Create folder
os.makedirs("reports/figures", exist_ok=True)

# Create plot
plt.figure(figsize=(12,6))

# Historical TSLA
plt.plot(
    range(len(tsla)),
    tsla["TSLA"],
    label="Historical TSLA",
    color="black"
)

# Forecast
plt.plot(
    forecast_index,
    arima["forecast"],
    label="ARIMA Forecast",
    color="blue"
)

# Confidence intervals
plt.fill_between(
    forecast_index,
    arima["lower_bound"],
    arima["upper_bound"],
    color="blue",
    alpha=0.2,
    label="Confidence Interval"
)

# Labels
plt.title("TSLA Price Forecast with Confidence Intervals")
plt.xlabel("Time Step")
plt.ylabel("TSLA Price")
plt.legend()
plt.grid(True)

# Save plot
plt.savefig(
    "reports/figures/tsla_arima_forecast.png",
    dpi=300,
    bbox_inches="tight"
)

print("Plot saved to reports/figures/tsla_arima_forecast.png")

# Show plot
plt.show()
