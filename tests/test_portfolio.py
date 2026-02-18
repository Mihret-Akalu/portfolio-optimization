# tests/test_portfolio.py

import numpy as np
import pandas as pd
import pytest
from src.portfolio import optimize_portfolio


def test_weights_sum_to_one():
    # Create sample returns data with fixed seed for reproducibility
    np.random.seed(42)
    n_samples = 252  # One year of trading days
    
    # Create realistic returns with controlled randomness
    returns_data = pd.DataFrame({
        'TSLA': np.random.normal(0.001, 0.02, n_samples),
        'SPY': np.random.normal(0.0005, 0.01, n_samples),
        'BND': np.random.normal(0.0002, 0.005, n_samples)
    })
    
    # Optimize portfolio
    weights = optimize_portfolio(returns_data)
    
    # Convert weights to array and check sum
    weight_values = np.array(list(weights.values()))
    assert abs(np.sum(weight_values) - 1.0) < 1e-6
    
    # Check we got weights for all assets
    assert set(weights.keys()) == {'TSLA', 'SPY', 'BND'}
    
    # Check weights are non-negative
    assert all(w >= 0 for w in weights.values())


def test_optimize_portfolio_edge_case():
    """Test with very low returns (should fall back to min variance)"""
    np.random.seed(42)
    
    # Create returns with very low mean
    returns_data = pd.DataFrame({
        'TSLA': np.random.normal(0.00001, 0.02, 252),
        'SPY': np.random.normal(0.00001, 0.01, 252),
        'BND': np.random.normal(0.00001, 0.005, 252)
    })
    
    # This should not raise an error
    weights = optimize_portfolio(returns_data)
    
    # Should still sum to 1
    weight_values = np.array(list(weights.values()))
    assert abs(np.sum(weight_values) - 1.0) < 1e-6


def test_optimize_portfolio_with_nan():
    """Test that function handles NaN values gracefully"""
    np.random.seed(42)
    
    # Create data with some NaN values
    returns_data = pd.DataFrame({
        'TSLA': np.random.normal(0.001, 0.02, 252),
        'SPY': np.random.normal(0.0005, 0.01, 252),
        'BND': np.random.normal(0.0002, 0.005, 252)
    })
    
    # Add some NaN values
    returns_data.iloc[10:15, 0] = np.nan
    
    # This should handle NaNs automatically
    weights = optimize_portfolio(returns_data)
    
    # Should still work
    weight_values = np.array(list(weights.values()))
    assert abs(np.sum(weight_values) - 1.0) < 1e-6


def test_optimize_portfolio_single_asset():
    """Test with single asset (should return 100% weight)"""
    np.random.seed(42)
    
    returns_data = pd.DataFrame({
        'TSLA': np.random.normal(0.001, 0.02, 252)
    })
    
    weights = optimize_portfolio(returns_data)
    
    assert len(weights) == 1
    assert abs(list(weights.values())[0] - 1.0) < 1e-6


def test_optimize_portfolio_equal_fallback():
    """Test that optimization falls back to equal weights when it fails"""
    np.random.seed(42)
    
    # Create pathological data that might cause optimization to fail
    returns_data = pd.DataFrame({
        'TSLA': [0.01] * 10 + [np.nan] * 242,  # Mostly NaN
        'SPY': [0.005] * 10 + [np.nan] * 242,
        'BND': [0.002] * 10 + [np.nan] * 242
    })
    
    # This should not raise an error
    weights = optimize_portfolio(returns_data)
    
    # Should return some weights (equal weights fallback)
    assert len(weights) == 3
    assert all(w > 0 for w in weights.values())