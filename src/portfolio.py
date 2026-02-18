# src/portfolio.py

import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import sample_cov
from pypfopt import expected_returns
import warnings


def optimize_portfolio(returns, risk_free_rate=0.02):
    """
    Optimize portfolio weights using mean-variance optimization
    
    Parameters:
    returns: DataFrame of asset returns
    risk_free_rate: annual risk-free rate (default 2%)
    
    Returns:
    dict: Optimal weights
    """
    # Drop any NaN values
    returns = returns.dropna()
    
    if returns.empty or len(returns.columns) < 2:
        warnings.warn("Insufficient data for optimization. Returning equal weights.")
        return {col: 1/len(returns.columns) for col in returns.columns}
    
    # Calculate expected returns and covariance
    try:
        mu = expected_returns.mean_historical_return(returns)
        S = sample_cov(returns)
        
        # Check for invalid values
        if mu.isnull().any() or np.isinf(mu).any():
            raise ValueError("Expected returns contain NaN or Inf")
        
        if S.isnull().any().any() or np.isinf(S).any().any():
            raise ValueError("Covariance matrix contains NaN or Inf")
        
    except Exception as e:
        warnings.warn(f"Error calculating statistics: {e}. Using equal weights.")
        return {col: 1/len(returns.columns) for col in returns.columns}
    
    try:
        # Try max Sharpe ratio
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        
    except Exception as e:
        if "at least one of the assets must have an expected return exceeding the risk-free rate" in str(e):
            # Fall back to min variance portfolio
            warnings.warn("No asset exceeds risk-free rate. Using minimum variance portfolio.")
            try:
                ef = EfficientFrontier(mu, S)
                weights = ef.min_volatility()
            except:
                # If all else fails, use equal weights
                warnings.warn("Optimization failed. Using equal weights.")
                return {col: 1/len(returns.columns) for col in returns.columns}
        else:
            # Try min variance as fallback for any other error
            warnings.warn(f"Max Sharpe failed ({str(e)}). Trying min variance.")
            try:
                ef = EfficientFrontier(mu, S)
                weights = ef.min_volatility()
            except:
                # Ultimate fallback: equal weights
                warnings.warn("Optimization failed. Using equal weights.")
                return {col: 1/len(returns.columns) for col in returns.columns}
    
    # Clean weights (round small values to zero)
    cleaned_weights = ef.clean_weights()
    
    return cleaned_weights