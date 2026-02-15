import pandas as pd
import os
from collections import OrderedDict

# example weights (replace if your script calculates them dynamically)
weights = OrderedDict([
    ('BND', 0.3006680605148493),
    ('SPY', 0.2083481312104325),
    ('TSLA', 0.4909838082747182)
])

# convert to dataframe
df = pd.DataFrame(list(weights.items()), columns=["Asset", "Weight"])

# create folder
os.makedirs("data/processed", exist_ok=True)

# save file
df.to_csv("data/processed/portfolio_weights.csv", index=False)

print("Portfolio weights saved.")
print(df)
