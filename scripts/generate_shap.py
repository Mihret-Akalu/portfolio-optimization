# scripts/generate_shap.py

import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from src.train_lstm import prepare_data, build_model

# Create folders
os.makedirs("reports/figures", exist_ok=True)

print("=" * 50)
print("SHAP Analysis for LSTM Model")
print("=" * 50)

# Generate synthetic data (replace with real returns later)
print("\n1. Generating synthetic data...")
np.random.seed(42)
series = pd.Series(np.random.randn(500).cumsum())
print(f"   Series length: {len(series)}")

# Prepare data
print("2. Preparing data...")
X, y, scaler = prepare_data(series, window=20)
print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")

# Build model FIRST (without input_shape parameter)
print("3. Building model...")
model = build_model()  # No parameters here!

# Then build the model by passing a single batch through it
# This is called "building" the model - it creates the weights
print(f"   Building model with input shape: {(None, X.shape[1], X.shape[2])}")
model.build(input_shape=(None, X.shape[1], X.shape[2]))
model.summary()  # This will now show the model architecture

# Train model
print("4. Training model...")
model.fit(X, y, epochs=5, batch_size=32, verbose=1)
print("   Model trained successfully")

# Use subset for SHAP (faster)
X_sample = X[:50]  # Use 50 samples
background = X[:25]  # Use 25 samples as background

print(f"5. Creating GradientExplainer with {len(background)} background samples...")
# Use GradientExplainer (better for TensorFlow 2.x)
explainer = shap.GradientExplainer(model, background)

print("6. Computing SHAP values...")
shap_values = explainer.shap_values(X_sample)

# Handle output shape
if isinstance(shap_values, list):
    print(f"   SHAP values is a list with {len(shap_values)} elements")
    shap_vals = shap_values[0]
else:
    shap_vals = shap_values

print(f"   SHAP values shape: {shap_vals.shape}")

# Reshape for visualization
shap_vals_2d = shap_vals.reshape(shap_vals.shape[0], -1)
X_2d = X_sample.reshape(X_sample.shape[0], -1)

# Create feature names
feature_names = [f't-{i}' for i in range(X.shape[1], 0, -1)]

print("7. Generating SHAP summary plot...")
plt.figure(figsize=(14, 10))
shap.summary_plot(
    shap_vals_2d,
    X_2d,
    feature_names=feature_names,
    show=False,
    max_display=20
)
plt.title('SHAP Feature Importance (GradientExplainer)', fontsize=16)
plt.tight_layout()

# Save figure
output_path = "reports/figures/shap_summary.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ SHAP summary plot saved to {output_path}")

print("\n✅ SHAP analysis complete!")