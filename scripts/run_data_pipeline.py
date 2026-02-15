from src.data_loader import download_price_data
from src.preprocessing import clean_price_data, compute_returns
from src.risk_metrics import compute_risk_metrics


def run():

    tickers = ["TSLA", "BND", "SPY"]

    data = download_price_data(
        tickers,
        "2015-01-01",
        "2026-01-15"
    )

    clean_data = clean_price_data(data)

    returns = compute_returns(clean_data)

    metrics = compute_risk_metrics(returns)

    clean_data.to_csv("data/processed/historical_prices.csv")

    returns.to_csv("data/processed/daily_returns.csv")

    metrics.to_csv("data/processed/risk_metrics.csv")

    print("Pipeline complete.")


if __name__ == "__main__":

    run()
