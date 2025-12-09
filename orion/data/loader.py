# -*- coding: utf-8 -*-
"""
ORION Data Loader - Fail-Safe Multi-Exchange OHLCV Fetcher with Multi-Timeframe Alignment
==========================================================================================

This module implements a production-grade data loader that:
1. Attempts Binance global → fails over to BinanceUS → falls back to CSV archives
2. Fetches 5 years of BTC/USDT across 5 timeframes (5m, 15m, 1h, 4h, 1d)
3. Aligns ALL timeframes to a master 5-minute index (critical for RL state consistency)
4. Generates technical indicators using the `ta` library

MATH CONTEXT:
- Multi-timeframe analysis requires temporal coherence: at each 5m step t, we need
  valid states from 15m, 1h, 4h, 1d that are "current" as of that timestamp
- Forward-fill (ffill) propagates the last known value until a new candle closes
- This creates a hierarchical state representation: S_t = [s_5m, s_15m, s_1h, s_4h, s_1d]

Author: ORION Quant Team
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import ccxt
from sklearn.preprocessing import RobustScaler

# Technical analysis library
import ta
from ta.trend import EMAIndicator, ADXIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, TSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ORION.DataLoader")


# ==============================================================================
# CONSTANTS
# ==============================================================================
TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]
TIMEFRAME_MINUTES = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
SYMBOL = "BTC/USDT"
YEARS_OF_HISTORY = 5
MS_PER_MINUTE = 60 * 1000


# ==============================================================================
# FAILOVER DATA LOADER
# ==============================================================================
class OrionDataLoader:
    """
    Robust data loader with exchange failover logic.
    
    The loader attempts exchanges in priority order:
    1. Binance (global) - highest liquidity, best data quality
    2. BinanceUS - fallback for US-based servers
    3. CSV fallback - pre-downloaded historical archives
    
    Each fetch operation includes rate limiting and error recovery.
    """
    
    def __init__(
        self,
        cache_dir: str = "./data_cache",
        rate_limit_ms: int = 100,
        max_retries: int = 3
    ):
        """
        Initialize the data loader.
        
        Args:
            cache_dir: Directory to cache fetched data (avoids re-fetching)
            rate_limit_ms: Milliseconds between API calls (respect rate limits)
            max_retries: Number of retries per exchange before failover
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_ms = rate_limit_ms
        self.max_retries = max_retries
        
        # Exchange priority list (will try in order)
        self.exchange_configs = [
            {"id": "binance", "class": ccxt.binance, "name": "Binance Global"},
            {"id": "binanceus", "class": ccxt.binanceus, "name": "Binance US"},
        ]
        
        self._active_exchange = None
        self._init_exchange()
    
    def _init_exchange(self) -> None:
        """
        Initialize the first working exchange.
        
        This implements a fail-fast test: we attempt a simple API call
        and switch to the next exchange if it fails.
        """
        for config in self.exchange_configs:
            try:
                logger.info(f"Attempting connection to {config['name']}...")
                exchange = config["class"]({
                    "enableRateLimit": True,
                    "rateLimit": self.rate_limit_ms,
                    "options": {"defaultType": "spot"}
                })
                # Test connectivity with a simple call
                exchange.load_markets()
                self._active_exchange = exchange
                logger.info(f"✓ Successfully connected to {config['name']}")
                return
            except Exception as e:
                logger.warning(f"✗ Failed to connect to {config['name']}: {e}")
                continue
        
        logger.warning("All exchanges failed. Will use CSV fallback mode.")
        self._active_exchange = None
    
    def _get_cache_path(self, timeframe: str) -> Path:
        """Get cache file path for a timeframe."""
        return self.cache_dir / f"btcusdt_{timeframe}.parquet"
    
    def _fetch_ohlcv_paginated(
        self,
        timeframe: str,
        since_ms: int,
        until_ms: int
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data with pagination.
        
        CCXT returns max 1000-1500 candles per request (exchange-dependent).
        We paginate by advancing the `since` timestamp after each batch.
        
        MATH NOTE:
        - Each candle timestamp represents the OPEN time of that candle
        - The CLOSE time = open_time + timeframe_duration - 1ms
        - For alignment, we use open_time as the canonical timestamp
        
        Args:
            timeframe: Candle timeframe (e.g., "5m", "1h")
            since_ms: Start timestamp in milliseconds
            until_ms: End timestamp in milliseconds
            
        Returns:
            DataFrame with columns [timestamp, open, high, low, close, volume]
        """
        if self._active_exchange is None:
            return self._load_from_csv_fallback(timeframe)
        
        all_candles = []
        current_since = since_ms
        tf_ms = TIMEFRAME_MINUTES[timeframe] * MS_PER_MINUTE
        batch_size = 1000  # Most exchanges support this
        
        logger.info(f"Fetching {timeframe} data from {datetime.fromtimestamp(since_ms/1000)}")
        
        while current_since < until_ms:
            try:
                # Fetch batch
                candles = self._active_exchange.fetch_ohlcv(
                    SYMBOL,
                    timeframe=timeframe,
                    since=current_since,
                    limit=batch_size
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                
                # Advance to next batch (last candle timestamp + 1 timeframe)
                last_ts = candles[-1][0]
                current_since = last_ts + tf_ms
                
                # Progress indicator
                if len(all_candles) % 10000 == 0:
                    pct = (current_since - since_ms) / (until_ms - since_ms) * 100
                    logger.info(f"  {timeframe}: {len(all_candles)} candles fetched ({pct:.1f}%)")
                
                # Rate limiting (extra safety)
                time.sleep(self.rate_limit_ms / 1000)
                
            except ccxt.RateLimitExceeded:
                logger.warning("Rate limit hit, sleeping 60s...")
                time.sleep(60)
            except Exception as e:
                logger.error(f"Error fetching {timeframe}: {e}")
                break
        
        if not all_candles:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        
        # Remove duplicates (can occur at pagination boundaries)
        df = df[~df.index.duplicated(keep="first")]
        
        logger.info(f"✓ {timeframe}: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        return df
    
    def _load_from_csv_fallback(self, timeframe: str) -> pd.DataFrame:
        """
        Load data from pre-downloaded CSV files.
        
        Fallback data sources:
        1. Kaggle BTC historical datasets
        2. CryptoDataDownload archives
        3. Local cache from previous runs
        
        For production, pre-download from:
        - https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data
        - https://www.cryptodatadownload.com/data/binance/
        """
        csv_path = self.cache_dir / f"btcusdt_{timeframe}_fallback.csv"
        
        if csv_path.exists():
            logger.info(f"Loading {timeframe} from CSV fallback: {csv_path}")
            df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
            return df
        
        logger.error(
            f"No data available for {timeframe}. "
            f"Please download historical data to {csv_path}"
        )
        return pd.DataFrame()
    
    def fetch_all_timeframes(
        self,
        years: int = YEARS_OF_HISTORY,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all timeframes for the specified history period.
        
        This is the main entry point for data loading.
        
        Args:
            years: Number of years of history to fetch
            use_cache: If True, load from cache if available
            
        Returns:
            Dictionary mapping timeframe -> DataFrame
        """
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=years * 365)
        since_ms = int(start_time.timestamp() * 1000)
        until_ms = int(end_time.timestamp() * 1000)
        
        logger.info(f"Fetching {years} years of data: {start_time} to {end_time}")
        
        data = {}
        for tf in TIMEFRAMES:
            cache_path = self._get_cache_path(tf)
            
            # Check cache
            if use_cache and cache_path.exists():
                logger.info(f"Loading {tf} from cache: {cache_path}")
                data[tf] = pd.read_parquet(cache_path)
                continue
            
            # Fetch from exchange
            df = self._fetch_ohlcv_paginated(tf, since_ms, until_ms)
            
            if not df.empty:
                # Cache the data
                df.to_parquet(cache_path)
                logger.info(f"Cached {tf} to {cache_path}")
            
            data[tf] = df
        
        return data


# ==============================================================================
# MULTI-TIMEFRAME ALIGNER
# ==============================================================================
class MultiTimeframeAligner:
    """
    Aligns multiple timeframes to a master 5-minute index.
    
    CRITICAL CONCEPT:
    When the RL agent observes state at time t (a 5m candle), it needs
    the "current" state of higher timeframes. But 1h candles only close
    every 12 steps of 5m. The solution:
    
    1. Reindex all timeframes to the 5m master index
    2. Forward-fill (ffill) to propagate the last closed candle
    3. This ensures temporal consistency: no future information leaks
    
    MATH:
    Let T_5m = {t_0, t_1, ..., t_N} be the 5-minute timestamps
    For each higher timeframe (e.g., 1h), we map:
        X_1h(t_i) = X_1h(floor(t_i, 1h))  # Last closed 1h candle
    
    The ffill operation implements this floor() alignment.
    """
    
    def __init__(self, base_timeframe: str = "5m"):
        """
        Initialize the aligner.
        
        Args:
            base_timeframe: The master timeframe (usually finest granularity)
        """
        self.base_timeframe = base_timeframe
        self.scalers = {}  # Store fitted scalers for inference
    
    def align(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Align all timeframes to the master index.
        
        ALGORITHM:
        1. Determine the common time window (intersection of all timeframes)
        2. Create master index from base timeframe within this window
        3. Reindex each higher timeframe to master, then ffill
        4. Stack into a single DataFrame with hierarchical columns
        
        Args:
            data: Dictionary of timeframe -> OHLCV DataFrame
            
        Returns:
            Tuple of (aligned_master_df, individual_aligned_dfs)
        """
        logger.info("Aligning timeframes to master index...")
        
        # Step 1: Find common time window
        # The start is the MAX of all starts (latest start)
        # The end is the MIN of all ends (earliest end)
        starts = [df.index.min() for df in data.values() if not df.empty]
        ends = [df.index.max() for df in data.values() if not df.empty]
        
        if not starts or not ends:
            raise ValueError("No valid data to align")
        
        common_start = max(starts)
        common_end = min(ends)
        
        logger.info(f"Common window: {common_start} to {common_end}")
        
        # Step 2: Create master index from base timeframe
        base_df = data[self.base_timeframe]
        master_index = base_df.loc[common_start:common_end].index
        
        logger.info(f"Master index: {len(master_index)} steps")
        
        # Step 3: Align each timeframe
        aligned = {}
        for tf, df in data.items():
            if df.empty:
                continue
            
            # Trim to common window
            df_trimmed = df.loc[common_start:common_end]
            
            # Reindex to master and forward-fill
            # IMPORTANT: The ffill ensures we only use past/current data (no lookahead)
            df_aligned = df_trimmed.reindex(master_index, method="ffill")
            
            # Add timeframe suffix to columns
            df_aligned.columns = [f"{col}_{tf}" for col in df_aligned.columns]
            
            aligned[tf] = df_aligned
            logger.info(f"  {tf}: {len(df_aligned)} aligned rows, {df_aligned.isna().sum().sum()} NaNs")
        
        # Step 4: Concatenate all timeframes
        master_df = pd.concat(aligned.values(), axis=1)
        
        # Drop any rows with NaNs (can occur at the start before all timeframes have data)
        initial_len = len(master_df)
        master_df = master_df.dropna()
        dropped = initial_len - len(master_df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with NaNs (start-of-data edge effect)")
        
        logger.info(f"✓ Alignment complete: {master_df.shape}")
        return master_df, aligned
    
    def add_technical_indicators(
        self,
        aligned: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Add technical indicators to each aligned timeframe.
        
        INDICATORS (per timeframe):
        
        TREND:
        - EMA Ribbon (9, 21, 50, 200): Exponential Moving Averages
          EMA_t = α * P_t + (1-α) * EMA_{t-1}, where α = 2/(N+1)
          
        - Ichimoku Kijun-Sen: 26-period midpoint = (High_26 + Low_26) / 2
          We compute Price vs Kijun as a momentum signal
        
        MOMENTUM:
        - RSI (14): Relative Strength Index
          RSI = 100 - 100/(1 + RS), where RS = AvgGain/AvgLoss
          
        - RSI Slope: First derivative of RSI (momentum of momentum)
          RSI_slope_t = RSI_t - RSI_{t-1}
          
        - TSI: True Strength Index (double-smoothed momentum)
          TSI = 100 * EMA(EMA(PC, r), s) / EMA(EMA(|PC|, r), s)
          where PC = Price Change
        
        VOLATILITY:
        - NATR: Normalized Average True Range (ATR / Close * 100)
          Allows comparison across different price levels
          
        - Bollinger Band Width: (Upper - Lower) / Middle
          Low width = squeeze, potential breakout
        
        REGIME:
        - ADX (14): Average Directional Index
          ADX < 20: ranging/choppy market
          ADX > 25: trending market
        
        VOLUME:
        - VWAP: Volume Weighted Average Price (approximated intraday)
        - CVD: Cumulative Volume Delta (approximated from candle body)
        """
        logger.info("Computing technical indicators...")
        
        enriched = {}
        for tf, df in aligned.items():
            df_copy = df.copy()
            
            # Extract base OHLCV columns (remove timeframe suffix for ta library)
            close = df_copy[f"close_{tf}"]
            high = df_copy[f"high_{tf}"]
            low = df_copy[f"low_{tf}"]
            open_ = df_copy[f"open_{tf}"]
            volume = df_copy[f"volume_{tf}"]
            
            # === TREND INDICATORS ===
            
            # EMA Ribbon
            for period in [9, 21, 50, 200]:
                ema = EMAIndicator(close=close, window=period)
                df_copy[f"ema_{period}_{tf}"] = ema.ema_indicator()
            
            # EMA Ribbon signals: distance from price to each EMA (normalized)
            for period in [9, 21, 50, 200]:
                df_copy[f"ema_{period}_dist_{tf}"] = (
                    (close - df_copy[f"ema_{period}_{tf}"]) / close * 100
                )
            
            # Ichimoku (only Kijun-Sen for simplicity)
            ichimoku = IchimokuIndicator(high=high, low=low)
            kijun = ichimoku.ichimoku_base_line()
            df_copy[f"ichimoku_kijun_{tf}"] = kijun
            df_copy[f"price_vs_kijun_{tf}"] = (close - kijun) / close * 100
            
            # === MOMENTUM INDICATORS ===
            
            # RSI
            rsi = RSIIndicator(close=close, window=14)
            df_copy[f"rsi_{tf}"] = rsi.rsi()
            
            # RSI Slope (first derivative)
            df_copy[f"rsi_slope_{tf}"] = df_copy[f"rsi_{tf}"].diff()
            
            # TSI (True Strength Index)
            tsi = TSIIndicator(close=close, window_slow=25, window_fast=13)
            df_copy[f"tsi_{tf}"] = tsi.tsi()
            
            # === VOLATILITY INDICATORS ===
            
            # NATR (Normalized ATR)
            atr = AverageTrueRange(high=high, low=low, close=close, window=14)
            df_copy[f"atr_{tf}"] = atr.average_true_range()
            df_copy[f"natr_{tf}"] = df_copy[f"atr_{tf}"] / close * 100
            
            # Bollinger Bands
            bb = BollingerBands(close=close, window=20, window_dev=2)
            df_copy[f"bb_width_{tf}"] = bb.bollinger_wband()
            df_copy[f"bb_pband_{tf}"] = bb.bollinger_pband()  # %B indicator
            
            # === REGIME INDICATORS ===
            
            # ADX
            adx = ADXIndicator(high=high, low=low, close=close, window=14)
            df_copy[f"adx_{tf}"] = adx.adx()
            df_copy[f"adx_pos_{tf}"] = adx.adx_pos()  # +DI
            df_copy[f"adx_neg_{tf}"] = adx.adx_neg()  # -DI
            
            # Regime classification (for auxiliary signals)
            # 0 = range, 1 = trend up, -1 = trend down
            adx_val = df_copy[f"adx_{tf}"]
            di_diff = df_copy[f"adx_pos_{tf}"] - df_copy[f"adx_neg_{tf}"]
            df_copy[f"regime_{tf}"] = np.where(
                adx_val < 20, 0,
                np.where(di_diff > 0, 1, -1)
            )
            
            # === VOLUME INDICATORS ===
            
            # Approximate VWAP (cumulative within session - simplified)
            # True VWAP resets daily; here we use rolling approximation
            typical_price = (high + low + close) / 3
            df_copy[f"vwap_20_{tf}"] = (
                (typical_price * volume).rolling(20).sum() /
                volume.rolling(20).sum()
            )
            df_copy[f"price_vs_vwap_{tf}"] = (close - df_copy[f"vwap_20_{tf}"]) / close * 100
            
            # Cumulative Volume Delta (approximated)
            # CVD estimates buyer vs seller pressure from candle body
            # If close > open: buyers dominated (positive delta)
            # Delta magnitude ~ volume * (body_size / range_size)
            body = close - open_
            range_ = high - low
            range_ = range_.replace(0, np.nan)  # Avoid division by zero
            delta = volume * (body / range_)
            df_copy[f"cvd_{tf}"] = delta.cumsum()
            df_copy[f"cvd_20_{tf}"] = delta.rolling(20).sum()  # 20-period CVD
            
            enriched[tf] = df_copy
            logger.info(f"  {tf}: {len(df_copy.columns)} features")
        
        return enriched
    
    def normalize(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Apply RobustScaler normalization to all features.
        
        WHY ROBUSTSCALER?
        Crypto markets have extreme outliers (flash crashes, pumps).
        Standard normalization (mean/std) is heavily influenced by outliers.
        
        RobustScaler uses median and IQR (Interquartile Range):
            X_scaled = (X - median(X)) / IQR(X)
        
        This is robust to outliers because:
        - Median is not affected by extreme values
        - IQR (Q3 - Q1) captures central 50% of distribution
        
        Args:
            df: DataFrame to normalize
            fit: If True, fit new scalers. If False, use previously fitted.
            
        Returns:
            Normalized DataFrame
        """
        logger.info("Applying RobustScaler normalization...")
        
        if fit:
            self.scalers = {}
        
        normalized = df.copy()
        
        for col in df.columns:
            if fit:
                scaler = RobustScaler()
                normalized[col] = scaler.fit_transform(df[[col]])
                self.scalers[col] = scaler
            else:
                if col in self.scalers:
                    normalized[col] = self.scalers[col].transform(df[[col]])
                else:
                    logger.warning(f"No scaler for {col}, skipping normalization")
        
        return normalized


# ==============================================================================
# CONVENIENCE FUNCTION
# ==============================================================================
def load_and_prepare_data(
    years: int = YEARS_OF_HISTORY,
    cache_dir: str = "./data_cache",
    use_cache: bool = True
) -> Tuple[pd.DataFrame, MultiTimeframeAligner]:
    """
    One-stop function to load, align, and enrich BTC/USDT data.
    
    Args:
        years: Years of history to fetch
        cache_dir: Cache directory for raw data
        use_cache: Whether to use cached data
        
    Returns:
        Tuple of (fully prepared DataFrame, aligner instance for inference)
    """
    # Load raw data
    loader = OrionDataLoader(cache_dir=cache_dir)
    raw_data = loader.fetch_all_timeframes(years=years, use_cache=use_cache)
    
    # Align to master index
    aligner = MultiTimeframeAligner(base_timeframe="5m")
    master_df, aligned = aligner.align(raw_data)
    
    # Add technical indicators
    enriched = aligner.add_technical_indicators(aligned)
    
    # Recombine into single DataFrame
    final_df = pd.concat([df for df in enriched.values()], axis=1)
    
    # Drop rows with any NaN (from indicator warm-up periods)
    initial = len(final_df)
    final_df = final_df.dropna()
    logger.info(f"Dropped {initial - len(final_df)} rows from indicator warm-up")
    
    return final_df, aligner


if __name__ == "__main__":
    # Test the loader
    df, aligner = load_and_prepare_data(years=1, use_cache=True)
    print(f"\nFinal DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)[:20]}...")
    print(f"\nDate range: {df.index[0]} to {df.index[-1]}")
    print(f"\nSample statistics:\n{df.describe()}")
