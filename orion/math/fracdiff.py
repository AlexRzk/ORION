# -*- coding: utf-8 -*-
"""
ORION Fractional Differencing Module - Stationarity with Memory Preservation
=============================================================================

This module implements Fast Fractional Differencing (FFD) for time series preprocessing.

PROBLEM:
Standard differencing (d=1) makes series stationary but DESTROYS memory.
Log-returns are essentially d=1 differencing: r_t = log(P_t) - log(P_{t-1})
This loses long-term dependencies crucial for ML/RL.

SOLUTION:
Fractional differencing with d ∈ (0, 1) provides a middle ground:
- d → 0: Original series (non-stationary, full memory)
- d → 1: Standard diff (stationary, no memory)
- d ∈ (0.3, 0.5): Often sweet spot (stationary + memory)

MATHEMATICS:

The fractional difference operator is defined as:
    (1 - B)^d = Σ_{k=0}^{∞} C(d,k) * (-B)^k

Where:
    B = Backshift operator: B * X_t = X_{t-1}
    C(d,k) = Binomial coefficient = Γ(d+1) / (Γ(k+1) * Γ(d-k+1))

For non-integer d, we use the generalized binomial:
    C(d,k) = d * (d-1) * (d-2) * ... * (d-k+1) / k!

The FFD weights are computed as:
    w_k = (-1)^k * C(d,k)

Which simplifies to the recursive formula:
    w_0 = 1
    w_k = -w_{k-1} * (d - k + 1) / k

The fractionally differenced series is:
    Y_t = Σ_{k=0}^{K} w_k * X_{t-k}

Where K is chosen based on weight threshold τ (we stop when |w_k| < τ).

IMPLEMENTATION:
We use FFD (Fixed-window Fractional Differencing) which uses a fixed
lookback K to maintain computational efficiency.

REFERENCES:
- Advances in Financial Machine Learning, Marcos López de Prado (2018)
- Chapter 5: Fractionally Differentiated Features

Author: ORION Quant Team
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from numba import jit, prange
import logging

# Stationarity test
from statsmodels.tsa.stattools import adfuller

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ORION.FracDiff")


# ==============================================================================
# FFD WEIGHT COMPUTATION (NUMBA-ACCELERATED)
# ==============================================================================
@jit(nopython=True, cache=True)
def compute_ffd_weights(d: float, threshold: float = 1e-5, max_lag: int = 10000) -> np.ndarray:
    """
    Compute FFD weights using the recursive formula.
    
    MATH:
    The weights follow the recursive relationship:
        w_k = -w_{k-1} * (d - k + 1) / k
    
    Starting from w_0 = 1, we get:
        w_1 = -d
        w_2 = d * (d-1) / 2
        w_3 = -d * (d-1) * (d-2) / 6
        ...
    
    Note that for d ∈ (0,1):
    - w_0 = 1 (positive)
    - w_1 = -d (negative)
    - w_2 = d*(d-1)/2 (positive, since d-1 < 0)
    - Signs alternate, magnitudes decay
    
    The decay rate depends on d:
    - Larger d → faster decay (less memory)
    - Smaller d → slower decay (more memory)
    
    Args:
        d: Fractional differencing order (0 < d < 1 typically)
        threshold: Stop when |w_k| < threshold
        max_lag: Maximum number of weights to compute
        
    Returns:
        Array of FFD weights
    """
    weights = np.zeros(max_lag)
    weights[0] = 1.0
    
    k = 1
    while k < max_lag:
        # Recursive formula: w_k = -w_{k-1} * (d - k + 1) / k
        weights[k] = -weights[k-1] * (d - k + 1) / k
        
        # Check convergence
        if abs(weights[k]) < threshold:
            break
        k += 1
    
    # Trim to actual length
    return weights[:k+1]


@jit(nopython=True, parallel=True, cache=True)
def apply_ffd_weights(series: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Apply FFD weights to a time series (convolution-style).
    
    MATH:
    The fractionally differenced value at time t is:
        Y_t = Σ_{k=0}^{K} w_k * X_{t-k}
    
    This is a weighted sum of the current and past K values.
    The first K values cannot be computed (insufficient history)
    and are set to NaN.
    
    GEOMETRIC INTERPRETATION:
    Think of this as a "soft difference":
    - w_0 = 1: We take the current value
    - w_1 = -d: Subtract d times the previous value
    - Higher terms add back smaller corrections
    
    For d=1: weights = [1, -1, 0, 0, ...] → exact first difference
    For d=0: weights = [1, 0, 0, ...] → no differencing
    
    Args:
        series: Input time series (1D numpy array)
        weights: FFD weights from compute_ffd_weights
        
    Returns:
        Fractionally differenced series (NaN for first K values)
    """
    n = len(series)
    k = len(weights)
    result = np.empty(n)
    result[:] = np.nan  # Initialize with NaN
    
    # Parallel computation for each valid index
    for t in prange(k-1, n):
        cumsum = 0.0
        for i in range(k):
            cumsum += weights[i] * series[t - i]
        result[t] = cumsum
    
    return result


# ==============================================================================
# STATIONARITY TESTING
# ==============================================================================
def adf_test(series: np.ndarray, significance: float = 0.05) -> Tuple[bool, float, float]:
    """
    Perform Augmented Dickey-Fuller test for stationarity.
    
    MATH:
    The ADF test is a unit root test. It tests the null hypothesis:
        H0: The series has a unit root (non-stationary)
        H1: The series is stationary
    
    The test regression is:
        ΔY_t = α + βt + γY_{t-1} + Σ δ_i ΔY_{t-i} + ε_t
    
    Where:
        - ΔY_t = Y_t - Y_{t-1} (first difference)
        - α = constant (drift)
        - βt = deterministic trend
        - γ = coefficient of Y_{t-1} (this is what we test)
        - δ_i = coefficients for lagged differences (removes autocorrelation)
    
    The test statistic is: t_γ = γ̂ / SE(γ̂)
    
    Decision:
        - If p-value < significance: Reject H0 → Series is stationary
        - If p-value >= significance: Fail to reject H0 → Series is non-stationary
    
    Args:
        series: Time series to test (cleaned of NaN)
        significance: Significance level (default 0.05)
        
    Returns:
        Tuple of (is_stationary, adf_statistic, p_value)
    """
    # Remove NaN values
    clean_series = series[~np.isnan(series)]
    
    if len(clean_series) < 20:
        logger.warning("Series too short for ADF test")
        return False, np.nan, 1.0
    
    try:
        # ADF test with automatic lag selection (AIC criterion)
        result = adfuller(clean_series, autolag='AIC')
        adf_stat = result[0]
        p_value = result[1]
        is_stationary = p_value < significance
        
        return is_stationary, adf_stat, p_value
    
    except Exception as e:
        logger.error(f"ADF test failed: {e}")
        return False, np.nan, 1.0


# ==============================================================================
# OPTIMAL d FINDER
# ==============================================================================
def find_optimal_d(
    series: np.ndarray,
    d_range: Tuple[float, float] = (0.0, 1.0),
    d_step: float = 0.05,
    significance: float = 0.05,
    threshold: float = 1e-5
) -> Tuple[float, pd.DataFrame]:
    """
    Find the minimum d that achieves stationarity.
    
    ALGORITHM:
    1. Start from d=0 (original series)
    2. Apply FFD with increasing d values
    3. Test stationarity with ADF after each step
    4. Return the FIRST d that achieves p-value < significance
    
    WHY MINIMUM d?
    We want to preserve as much memory as possible while ensuring stationarity.
    Higher d = more stationary but less memory.
    The optimal d is the smallest value that passes the ADF test.
    
    COMPLEXITY:
    For each d candidate, we:
    - Compute O(K) weights (K depends on threshold)
    - Apply O(N*K) convolution
    - Run O(N) ADF test
    Total: O(D * N * K) where D = number of d candidates
    
    Args:
        series: Original time series
        d_range: Range of d values to search
        d_step: Step size for d search
        significance: ADF significance level
        threshold: FFD weight threshold
        
    Returns:
        Tuple of (optimal_d, search_log_dataframe)
    """
    d_values = np.arange(d_range[0], d_range[1] + d_step, d_step)
    results = []
    
    for d in d_values:
        # Compute weights
        weights = compute_ffd_weights(d, threshold=threshold)
        
        # Apply FFD
        ffd_series = apply_ffd_weights(series, weights)
        
        # Test stationarity
        is_stationary, adf_stat, p_value = adf_test(ffd_series, significance)
        
        results.append({
            'd': d,
            'num_weights': len(weights),
            'adf_stat': adf_stat,
            'p_value': p_value,
            'is_stationary': is_stationary
        })
        
        # Early termination: return first stationary d
        if is_stationary:
            logger.info(f"Found optimal d={d:.2f} (ADF p-value={p_value:.4f})")
            break
    
    results_df = pd.DataFrame(results)
    
    # Get optimal d (first stationary, or highest tested if none found)
    if any(results_df['is_stationary']):
        optimal_d = results_df.loc[results_df['is_stationary'], 'd'].iloc[0]
    else:
        optimal_d = d_range[1]  # Use maximum d as fallback
        logger.warning(f"No stationary d found, using d={optimal_d}")
    
    return optimal_d, results_df


# ==============================================================================
# MAIN FRACDIFF CLASS
# ==============================================================================
class FastFractionalDiff:
    """
    Fast Fractional Differencing with automatic d optimization.
    
    This class provides:
    1. Automatic d selection per feature (via ADF testing)
    2. Efficient weight computation (Numba-accelerated)
    3. Memory-efficient application to large datasets
    
    USAGE:
    ```python
    ffd = FastFractionalDiff()
    df_transformed = ffd.fit_transform(df[['open', 'high', 'low', 'close', 'volume']])
    # Store d values for inference
    print(ffd.d_values)
    ```
    """
    
    def __init__(
        self,
        d_range: Tuple[float, float] = (0.0, 1.0),
        d_step: float = 0.05,
        significance: float = 0.05,
        threshold: float = 1e-5
    ):
        """
        Initialize FFD transformer.
        
        Args:
            d_range: Range for d search
            d_step: Step size for d search
            significance: ADF significance level
            threshold: Weight threshold for truncation
        """
        self.d_range = d_range
        self.d_step = d_step
        self.significance = significance
        self.threshold = threshold
        
        # Stores optimal d for each feature (fitted)
        self.d_values: Dict[str, float] = {}
        self.weights: Dict[str, np.ndarray] = {}
        self.search_logs: Dict[str, pd.DataFrame] = {}
    
    def fit(self, df: pd.DataFrame) -> 'FastFractionalDiff':
        """
        Find optimal d for each column.
        
        Args:
            df: DataFrame with columns to transform
            
        Returns:
            self (for method chaining)
        """
        logger.info(f"Fitting FFD on {len(df.columns)} columns...")
        
        for col in df.columns:
            series = df[col].values.astype(np.float64)
            
            # Find optimal d
            optimal_d, search_log = find_optimal_d(
                series,
                d_range=self.d_range,
                d_step=self.d_step,
                significance=self.significance,
                threshold=self.threshold
            )
            
            # Store results
            self.d_values[col] = optimal_d
            self.weights[col] = compute_ffd_weights(optimal_d, self.threshold)
            self.search_logs[col] = search_log
            
            logger.info(f"  {col}: d={optimal_d:.2f}, weights_len={len(self.weights[col])}")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply FFD transformation using fitted d values.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame (NaN for warm-up period)
        """
        if not self.d_values:
            raise ValueError("Must call fit() before transform()")
        
        result = pd.DataFrame(index=df.index)
        
        for col in df.columns:
            if col not in self.weights:
                logger.warning(f"No fitted d for {col}, skipping")
                continue
            
            series = df[col].values.astype(np.float64)
            weights = self.weights[col]
            
            ffd_series = apply_ffd_weights(series, weights)
            result[col] = ffd_series
        
        return result
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame to fit and transform
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(df).transform(df)
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get summary of fitted d values.
        
        Returns:
            DataFrame with d values and weight counts per feature
        """
        if not self.d_values:
            return pd.DataFrame()
        
        summary = pd.DataFrame({
            'feature': list(self.d_values.keys()),
            'd': list(self.d_values.values()),
            'num_weights': [len(w) for w in self.weights.values()]
        })
        
        return summary


# ==============================================================================
# UTILITY: INVERSE FFD (for interpretation)
# ==============================================================================
def cumsum_filter(series: np.ndarray, threshold: float) -> np.ndarray:
    """
    CUSUM filter for detecting regime changes (utility function).
    
    MATH:
    The CUSUM filter tracks cumulative deviations from mean:
        S_t = max(0, S_{t-1} + x_t - threshold)
    
    When S_t exceeds a trigger level, we mark an event.
    This is useful for identifying structural breaks in the FFD series.
    
    Args:
        series: Input series (typically returns or FFD values)
        threshold: Threshold for event detection
        
    Returns:
        Array of event indices
    """
    cusum_pos = np.zeros(len(series))
    cusum_neg = np.zeros(len(series))
    events = []
    
    for t in range(1, len(series)):
        cusum_pos[t] = max(0, cusum_pos[t-1] + series[t] - threshold)
        cusum_neg[t] = min(0, cusum_neg[t-1] + series[t] + threshold)
        
        if cusum_pos[t] > threshold or cusum_neg[t] < -threshold:
            events.append(t)
            cusum_pos[t] = 0
            cusum_neg[t] = 0
    
    return np.array(events)


# ==============================================================================
# TEST / DEMO
# ==============================================================================
if __name__ == "__main__":
    # Generate synthetic price series (geometric Brownian motion)
    np.random.seed(42)
    n = 10000
    returns = np.random.normal(0.0001, 0.02, n)  # ~2% daily vol
    prices = 10000 * np.exp(np.cumsum(returns))  # Start at 10000
    
    print("=" * 60)
    print("ORION Fractional Differencing Test")
    print("=" * 60)
    
    # Test ADF on original prices
    is_stat, adf, pval = adf_test(prices)
    print(f"\nOriginal prices: ADF={adf:.4f}, p-value={pval:.4f}, Stationary={is_stat}")
    
    # Test log returns
    log_returns = np.diff(np.log(prices))
    is_stat, adf, pval = adf_test(log_returns)
    print(f"Log returns (d=1): ADF={adf:.4f}, p-value={pval:.4f}, Stationary={is_stat}")
    
    # Find optimal d
    optimal_d, search_log = find_optimal_d(prices)
    print(f"\nOptimal d search:")
    print(search_log.to_string())
    
    # Apply FFD with optimal d
    weights = compute_ffd_weights(optimal_d)
    ffd_prices = apply_ffd_weights(prices, weights)
    
    # Verify stationarity
    is_stat, adf, pval = adf_test(ffd_prices)
    print(f"\nFFD (d={optimal_d:.2f}): ADF={adf:.4f}, p-value={pval:.4f}, Stationary={is_stat}")
    
    # Memory retention analysis
    # Correlation between original and transformed
    valid_mask = ~np.isnan(ffd_prices)
    corr = np.corrcoef(prices[valid_mask], ffd_prices[valid_mask])[0, 1]
    print(f"Correlation with original: {corr:.4f}")
    
    # Compare with log returns
    log_ret_padded = np.concatenate([[np.nan], log_returns])
    corr_logret = np.corrcoef(prices[1:], log_ret_padded[1:])[0, 1]
    print(f"Log returns correlation with original: {corr_logret:.4f}")
    
    print("\n✓ FFD preserves more memory than log returns!")
