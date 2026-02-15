import pandas as pd
import matplotlib.pyplot as plt
import os

# load weights
df = pd.read_csv("data/processed/portfolio_weights.csv")

# create folder
os.makedirs("reports/figures", exist_ok=True)

# plot
plt.figure(figsize=(8,8))

plt.pie(
    df["Weight"],
    labels=df["Asset"],
    autopct="%1.1f%%",
)

plt.title("Optimal Portfolio Allocation")

# save
plt.savefig(
    "reports/figures/portfolio_allocation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Portfolio allocation plot saved.")
