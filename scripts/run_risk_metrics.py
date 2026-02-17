import pandas as pd
from src.risk_metrics import compute_risk_metrics

returns = pd.read_csv(
    "data/processed/daily_returns.csv",
    index_col="Date",
    parse_dates=True
)

metrics = compute_risk_metrics(returns)

metrics.to_csv(
    "data/processed/risk_metrics.csv"
)

print("Risk metrics saved.")
