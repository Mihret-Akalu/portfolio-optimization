# scripts/generate_shap.py

import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from src.train_lstm import prepare_data


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/processed/historical_prices.csv"
MODEL_PATH = "models/lstm_model.h5"
OUTPUT_DIR = "reports/figures"
WINDOW_SIZE = 60  # CHANGED: Match the model's expected input shape
SHAP_SAMPLE_SIZE = 50  # Reduced for faster computation
SHAP_BACKGROUND_SIZE = 30  # Reduced for faster computation

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("SHAP Explainability Analysis for LSTM Model")
print("=" * 60)


# ============================================================
# Step 1 — Load real financial data
# ============================================================

print("\n1. Loading historical price data...")

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["Date"],
    index_col="Date"
)

if "TSLA" not in df.columns:
    raise ValueError("TSLA column not found in historical_prices.csv")

# Use RETURNS (professional finance practice)
series = df["TSLA"].pct_change().dropna()

print(f"Series length: {len(series)}")
print(f"Date range: {series.index.min()} → {series.index.max()}")


# ============================================================
# Step 2 — Prepare LSTM input data with CORRECT window size
# ============================================================

print(f"\n2. Preparing LSTM input sequences (window={WINDOW_SIZE})...")

X, y, scaler = prepare_data(series, window=WINDOW_SIZE)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


# ============================================================
# Step 3 — Load trained LSTM model
# ============================================================

print("\n3. Loading trained LSTM model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"LSTM model not found at {MODEL_PATH}. "
        f"Run training first: python -m scripts.run_train_models"
    )

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully")
model.summary()

# Verify input shape
expected_shape = model.input_shape
print(f"\nModel expects input shape: {expected_shape}")
print(f"Data has shape: {X.shape}")
if expected_shape[1] != X.shape[1]:
    print(f"⚠ WARNING: Shape mismatch! Model expects {expected_shape[1]} time steps, got {X.shape[1]}")


# ============================================================
# Step 4 — Select SHAP samples
# ============================================================

print("\n4. Selecting SHAP background and evaluation samples...")

# Ensure we have enough samples
if len(X) < SHAP_BACKGROUND_SIZE + SHAP_SAMPLE_SIZE:
    SHAP_SAMPLE_SIZE = min(20, len(X) // 3)
    SHAP_BACKGROUND_SIZE = min(20, len(X) // 3)
    print(f"   Adjusted sample sizes: background={SHAP_BACKGROUND_SIZE}, eval={SHAP_SAMPLE_SIZE}")

background = X[:SHAP_BACKGROUND_SIZE]
X_sample = X[SHAP_BACKGROUND_SIZE:SHAP_BACKGROUND_SIZE + SHAP_SAMPLE_SIZE]

print(f"Background samples: {background.shape}")
print(f"Evaluation samples: {X_sample.shape}")


# ============================================================
# Step 5 — Create SHAP explainer
# ============================================================

print("\n5. Creating SHAP GradientExplainer...")

explainer = shap.GradientExplainer(model, background)


# ============================================================
# Step 6 — Compute SHAP values
# ============================================================

print("\n6. Computing SHAP values (this may take a few minutes)...")

try:
    shap_values = explainer.shap_values(X_sample)
    
    # Handle output format
    if isinstance(shap_values, list):
        print(f"   SHAP values is a list with {len(shap_values)} elements")
        shap_vals = shap_values[0]
    else:
        shap_vals = shap_values
    
    print(f"   SHAP values shape: {shap_vals.shape}")
    
except Exception as e:
    print(f"   Error computing SHAP values: {e}")
    print("\n   Trying with a smaller sample...")
    
    # Try with even smaller sample
    X_tiny = X_sample[:10]
    shap_values = explainer.shap_values(X_tiny)
    
    if isinstance(shap_values, list):
        shap_vals = shap_values[0]
    else:
        shap_vals = shap_values
    
    print(f"   SHAP values shape: {shap_vals.shape}")
    X_sample = X_tiny


# ============================================================
# Step 7 — Reshape for visualization
# ============================================================

print("\n7. Preparing SHAP visualization...")

shap_vals_2d = shap_vals.reshape(shap_vals.shape[0], -1)
X_2d = X_sample.reshape(X_sample.shape[0], -1)

# Create feature names for each lag
feature_names = [f"t-{i}" for i in range(WINDOW_SIZE, 0, -1)]


# ============================================================
# Step 8 — Generate SHAP summary plot
# ============================================================

print("\n8. Generating SHAP summary plot...")

plt.figure(figsize=(14, 10))

# Limit display to top features for readability
max_display = min(20, WINDOW_SIZE)

shap.summary_plot(
    shap_vals_2d,
    X_2d,
    feature_names=feature_names,
    show=False,
    max_display=max_display
)

plt.title(
    f"SHAP Feature Importance for LSTM Forecast Model (Window={WINDOW_SIZE})",
    fontsize=16
)

plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "shap_summary.png")

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"\n✓ SHAP summary saved to: {output_path}")


# ============================================================
# Step 9 — Generate bar plot of top features
# ============================================================

print("\n9. Generating feature importance bar plot...")

plt.figure(figsize=(12, 8))
mean_abs_shap = np.mean(np.abs(shap_vals_2d), axis=0)
top_n = min(15, len(mean_abs_shap))
top_indices = np.argsort(mean_abs_shap)[-top_n:]

plt.barh(range(len(top_indices)), mean_abs_shap[top_indices])
plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
plt.xlabel('Mean |SHAP value| (impact on model output)')
plt.title(f'Top {top_n} Feature Importances from SHAP Analysis', fontsize=16)
plt.tight_layout()

bar_path = os.path.join(OUTPUT_DIR, "shap_feature_importance.png")
plt.savefig(bar_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ Feature importance plot saved to: {bar_path}")


# ============================================================
# Step 10 — Interpretation guide
# ============================================================

print("\n" + "=" * 60)
print("SHAP Analysis Complete!")
print("=" * 60)
print("\nInterpretation Guide:")
print("• Higher SHAP value magnitude = stronger influence on prediction")
print("• Positive SHAP value = increases the predicted price")
print("• Negative SHAP value = decreases the predicted price")
print("• Earlier lags (t-60 to t-30) show long-term influence")
print("• Recent lags (t-30 to t-1) show short-term influence")
print("\nFiles saved:")
print(f"  • {output_path}")
print(f"  • {bar_path}")
print("=" * 60)