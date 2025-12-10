# -*- coding: utf-8 -*-
"""
ORION Training & Backtesting Pipeline
======================================

This module implements the complete training loop for the TFT + QR-DQN agent,
optimized for RTX 4090 GPUs on Vast.ai infrastructure.

KEY FEATURES:
1. Mixed Precision Training (FP16) for maximum batch size
2. Cosine Annealing with Warm Restarts scheduler
3. Sortino Ratio-based model checkpointing (not just loss!)
4. Experience Replay with Prioritized sampling
5. Target Network updates (Polyak averaging)
6. Comprehensive backtesting with transaction costs

TRAINING PIPELINE:
1. Load & preprocess data (FFD, normalization)
2. Split: 70% train, 15% validation, 15% test
3. Train with replay buffer and target network
4. Validate every epoch, save best Sortino model
5. Final backtest on held-out test set

PERFORMANCE METRICS:
- Sortino Ratio: Risk-adjusted return using downside deviation
  Sortino = (R - Rf) / σ_d
  Where σ_d = sqrt(E[min(R-Rf, 0)²])
  
- Sharpe Ratio: (R - Rf) / σ
- Max Drawdown: Maximum peak-to-trough decline
- Win Rate: % of profitable trades
- Profit Factor: Gross Profit / Gross Loss

Author: ORION Quant Team
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Enable CUDA memory optimization to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Local imports
from orion.data.loader import OrionDataLoader, MultiTimeframeAligner, load_and_prepare_data
from orion.math.fracdiff import FastFractionalDiff
from orion.models.hybrid import TFT_QR_DQN, create_model

# Enable PyTorch performance optimizations
torch.backends.cudnn.benchmark = True  # Auto-tune kernels for this input size
torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for faster matmul on Ampere GPUs
torch.backends.cudnn.allow_tf32 = True

# Configure logger only if not already configured (prevents duplicate logs)
logger = logging.getLogger("ORION.Train")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent duplicate logs from root logger


# ==============================================================================
# RANGER OPTIMIZER (RAdam + Lookahead)
# ==============================================================================
class Ranger(torch.optim.Optimizer):
    """
    Ranger Optimizer: RAdam + Lookahead.
    
    COMPONENTS:
    
    1. RAdam (Rectified Adam):
       - Fixes the variance issue in early training
       - No need for learning rate warmup
       - ρ_∞ = 2/(1-β₂) - 1
       - ρ_t = ρ_∞ - 2*t*β₂^t / (1 - β₂^t)
       - r_t = sqrt((ρ_t - 4)(ρ_t - 2)ρ_∞ / ((ρ_∞ - 4)(ρ_∞ - 2)ρ_t))
    
    2. Lookahead:
       - Maintains "slow" weights that explore the loss surface
       - Every k steps: slow = slow + α * (fast - slow)
       - More stable than vanilla optimizers
    
    WHY FOR TRADING?
    - More stable training with financial time series
    - Less sensitivity to hyperparameters
    - Better generalization (critical for out-of-sample)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        # Lookahead params
        k: int = 6,
        alpha: float = 0.5
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            k=k, alpha=alpha
        )
        super().__init__(params, defaults)
        
        # Lookahead state
        self.k = k
        self.alpha = alpha
        self._step_count = 0
        
        # Initialize slow weights
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['slow_buffer'] = p.data.clone()
    
    @torch.no_grad()
    def step(self, closure=None):
        """Single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Ranger does not support sparse gradients")
                
                state = self.state[p]
                
                # State initialization
                if 'step' not in state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                step = state['step']
                
                # Decoupled weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
                
                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                # RAdam: compute adaptive learning rate
                rho_inf = 2.0 / (1 - beta2) - 1
                rho_t = rho_inf - 2 * step * (beta2 ** step) / bias_correction2
                
                if rho_t > 5:
                    # Variance is tractable
                    rect = ((rho_t - 4) * (rho_t - 2) * rho_inf /
                            ((rho_inf - 4) * (rho_inf - 2) * rho_t)) ** 0.5
                    
                    step_size = group['lr'] * rect / bias_correction1
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    # Not enough variance, use unadapted step
                    step_size = group['lr'] / bias_correction1
                    p.data.add_(exp_avg, alpha=-step_size)
        
        # Lookahead step
        self._step_count += 1
        if self._step_count >= self.k:
            self._step_count = 0
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    slow = state['slow_buffer']
                    # slow = slow + alpha * (fast - slow)
                    slow.add_(p.data - slow, alpha=self.alpha)
                    p.data.copy_(slow)
        
        return loss


# ==============================================================================
# EXPERIENCE REPLAY BUFFER
# ==============================================================================
class ReplayBuffer:
    """
    Experience Replay Buffer with uniform sampling.
    
    MATH:
    The buffer stores transitions (s, a, r, s', done).
    
    For training stability:
    - Breaks temporal correlation between samples
    - Each transition can be used multiple times
    - Maintains a sliding window of recent experience
    
    The buffer is a circular queue with maximum size N.
    When full, oldest experiences are replaced.
    
    For Vast.ai with 24GB VRAM:
    - State size ≈ 96 steps × 200 features × 4 bytes = 77KB per state
    - 100K transitions ≈ 15GB (we store on CPU, transfer batches to GPU)
    """
    
    def __init__(
        self,
        capacity: int = 100_000,
        state_shape: Tuple[int, int] = (96, 200),
        device: str = 'cpu'
    ):
        """
        Args:
            capacity: Maximum buffer size
            state_shape: Shape of state (T, F)
            device: Device to store buffer (usually CPU)
        """
        self.capacity = capacity
        self.state_shape = state_shape
        self.device = device
        
        self.position = 0
        self.size = 0
        
        # Pre-allocate tensors (more efficient than list)
        T, F = state_shape
        self.states = torch.zeros((capacity, T, F), dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, T, F), dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)
    
    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool
    ) -> None:
        """Add a transition to the buffer."""
        idx = self.position
        
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(
        self,
        batch_size: int,
        to_device: str = 'cuda'
    ) -> Tuple[torch.Tensor, ...]:
        """
        Sample a random batch.
        
        Args:
            batch_size: Number of transitions
            to_device: Device to transfer batch to
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        indices = torch.randint(0, self.size, (batch_size,))
        
        return (
            self.states[indices].to(to_device),
            self.actions[indices].to(to_device),
            self.rewards[indices].to(to_device),
            self.next_states[indices].to(to_device),
            self.dones[indices].to(to_device)
        )
    
    def __len__(self) -> int:
        return self.size


# ==============================================================================
# TRADING ENVIRONMENT
# ==============================================================================
class TradingEnvironment:
    """
    RL Environment for Bitcoin trading.
    
    STATE:
    Multi-timeframe features window (T timesteps × F features)
    
    ACTION SPACE:
    0 = Long:  Enter long position (or hold if already long)
    1 = Short: Enter short position (or hold if already short)
    2 = Hold:  Maintain current position (no trade)
    3 = Close: Close any open position
    
    REWARD:
    PnL from the action, accounting for:
    - Price change (main component)
    - Transaction costs (spread + commission)
    - Slippage (function of volatility)
    
    MATH:
    For going Long at price P_t:
        Position = +1
        Entry price = P_t * (1 + spread/2 + commission)
    
    For closing Long at price P_{t+k}:
        Exit price = P_{t+k} * (1 - spread/2 - commission)
        PnL = (Exit - Entry) / Entry
    
    Reward at each step = position * (P_t - P_{t-1}) / P_{t-1} - costs
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        lookback: int = 96,
        initial_balance: float = 100_000.0,
        transaction_cost: float = 0.001,  # 0.1% per trade (spread + commission)
        max_position: float = 1.0
    ):
        """
        Args:
            data: Preprocessed DataFrame with all features
            lookback: Number of past steps to include in state
            initial_balance: Starting capital
            transaction_cost: Cost per trade (fraction)
            max_position: Maximum position size (1 = 100% of capital)
        """
        self.data = data
        self.lookback = lookback
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        
        # Extract close prices for reward calculation
        # CRITICAL: Use raw prices, NOT FFD-transformed values!
        # FFD values are stationary and don't represent actual prices
        raw_close_col = [c for c in data.columns if '_raw_close_5m' in c.lower()]
        if raw_close_col:
            self.prices = data[raw_close_col[0]].values
            logger.info(f"TradingEnvironment using raw close prices from '{raw_close_col[0]}'")
        else:
            # Fallback: Try to find close_5m (may be FFD-transformed - will warn)
            close_col = [c for c in data.columns if 'close_5m' in c.lower()]
            if close_col:
                self.prices = data[close_col[0]].values
                logger.warning(f"Using '{close_col[0]}' for prices - may be FFD-transformed!")
            else:
                # Last fallback
                close_cols = [c for c in data.columns if 'close' in c.lower()]
                self.prices = data[close_cols[0]].values if close_cols else data.iloc[:, 3].values
                logger.warning("Using fallback close column for prices")
        
        # === SANITY CHECK: Verify prices look like actual BTC prices ===
        price_min, price_max = self.prices.min(), self.prices.max()
        price_mean = self.prices.mean()
        logger.info(f"Price sanity check: min=${price_min:.2f}, max=${price_max:.2f}, mean=${price_mean:.2f}")
        
        if price_max < 100 or price_min < 0:
            logger.error(f"⚠ PRICE ANOMALY DETECTED! Prices don't look like BTC/USD.")
            logger.error(f"  This likely means FFD-transformed or normalized values are being used.")
            logger.error(f"  Expected: $10,000 - $100,000 range for BTC")
            logger.error(f"  Got: ${price_min:.4f} - ${price_max:.4f}")
            # Don't raise - allow training to continue but warn heavily
        
        # Feature matrix
        self.features = data.values.astype(np.float32)
        
        # Episode tracking
        self.current_step = 0
        self.position = 0  # -1, 0, or 1
        self.entry_price = 0.0
        self.balance = initial_balance
        self.max_steps = len(data) - lookback
        
        # Performance tracking
        self.trades = []
        self.equity_curve = []
    
    def reset(self) -> torch.Tensor:
        """Reset environment to start of episode."""
        self.current_step = self.lookback
        self.position = 0
        self.entry_price = 0.0
        self.balance = self.initial_balance
        self.trades = []
        self.equity_curve = [self.initial_balance]
        
        return self._get_state()
    
    def _get_state(self) -> torch.Tensor:
        """Get current state (lookback window of features)."""
        start = self.current_step - self.lookback
        end = self.current_step
        state = self.features[start:end]
        return torch.from_numpy(state)
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Execute action and return (next_state, reward, done, info).
        
        REWARD CALCULATION:
        1. Base reward = position * price_change_pct
        2. Transaction cost deducted on position changes
        3. Holding cost (optional, for leverage)
        """
        current_price = self.prices[self.current_step]
        prev_price = self.prices[self.current_step - 1]
        price_return = (current_price - prev_price) / prev_price
        
        reward = 0.0
        trade_executed = False
        
        # === ACTION LOGIC ===
        
        if action == 0:  # Long
            if self.position == 0:
                # Enter long
                self.position = 1
                self.entry_price = current_price * (1 + self.transaction_cost)
                reward -= self.transaction_cost  # Cost to enter
                trade_executed = True
            elif self.position == -1:
                # Close short, enter long
                exit_price = current_price * (1 + self.transaction_cost)
                pnl = (self.entry_price - exit_price) / self.entry_price
                reward += pnl - self.transaction_cost
                self.trades.append({
                    'type': 'short', 'pnl': pnl,
                    'entry': self.entry_price, 'exit': exit_price
                })
                
                self.position = 1
                self.entry_price = current_price * (1 + self.transaction_cost)
                reward -= self.transaction_cost
                trade_executed = True
            else:
                # Already long, hold
                reward = price_return
        
        elif action == 1:  # Short
            if self.position == 0:
                # Enter short
                self.position = -1
                self.entry_price = current_price * (1 - self.transaction_cost)
                reward -= self.transaction_cost
                trade_executed = True
            elif self.position == 1:
                # Close long, enter short
                exit_price = current_price * (1 - self.transaction_cost)
                pnl = (exit_price - self.entry_price) / self.entry_price
                reward += pnl - self.transaction_cost
                self.trades.append({
                    'type': 'long', 'pnl': pnl,
                    'entry': self.entry_price, 'exit': exit_price
                })
                
                self.position = -1
                self.entry_price = current_price * (1 - self.transaction_cost)
                reward -= self.transaction_cost
                trade_executed = True
            else:
                # Already short, hold
                reward = -price_return
        
        elif action == 2:  # Hold
            if self.position == 1:
                reward = price_return
            elif self.position == -1:
                reward = -price_return
            # If position == 0, reward = 0
        
        elif action == 3:  # Close
            if self.position == 1:
                exit_price = current_price * (1 - self.transaction_cost)
                pnl = (exit_price - self.entry_price) / self.entry_price
                reward += pnl - self.transaction_cost
                self.trades.append({
                    'type': 'long', 'pnl': pnl,
                    'entry': self.entry_price, 'exit': exit_price
                })
                self.position = 0
                self.entry_price = 0.0
                trade_executed = True
            elif self.position == -1:
                exit_price = current_price * (1 + self.transaction_cost)
                pnl = (self.entry_price - exit_price) / self.entry_price
                reward += pnl - self.transaction_cost
                self.trades.append({
                    'type': 'short', 'pnl': pnl,
                    'entry': self.entry_price, 'exit': exit_price
                })
                self.position = 0
                self.entry_price = 0.0
                trade_executed = True
        
        # Update balance
        self.balance *= (1 + reward)
        self.equity_curve.append(self.balance)
        
        # Advance step
        self.current_step += 1
        done = self.current_step >= self.max_steps + self.lookback - 1
        
        # Get next state
        if not done:
            next_state = self._get_state()
        else:
            next_state = torch.zeros((self.lookback, self.features.shape[1]))
        
        info = {
            'position': self.position,
            'balance': self.balance,
            'trade_executed': trade_executed,
            'price': current_price
        }
        
        return next_state, reward, done, info
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics from episode."""
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]
        
        if len(returns) == 0:
            return {}
        
        # Total return
        total_return = (equity[-1] - equity[0]) / equity[0]
        
        # Sharpe Ratio (annualized, assuming 5m bars → 288 per day × 365)
        periods_per_year = 288 * 365
        avg_return = returns.mean() * periods_per_year
        std_return = returns.std() * np.sqrt(periods_per_year)
        sharpe = avg_return / std_return if std_return > 0 else 0
        
        # Sortino Ratio (only downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std() * np.sqrt(periods_per_year)
            sortino = avg_return / downside_std if downside_std > 0 else 0
        else:
            sortino = float('inf') if avg_return > 0 else 0
        
        # Max Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = drawdown.max()
        
        # Win Rate and Profit Factor
        if self.trades:
            pnls = [t['pnl'] for t in self.trades]
            wins = sum(1 for p in pnls if p > 0)
            win_rate = wins / len(pnls)
            
            gross_profit = sum(p for p in pnls if p > 0)
            gross_loss = abs(sum(p for p in pnls if p < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            win_rate = 0
            profit_factor = 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': len(self.trades)
        }


# ==============================================================================
# MODEL SAVER (SORTINO-BASED)
# ==============================================================================
class ModelSaver:
    """
    Model checkpointing based on Validation Sortino Ratio.
    
    WHY SORTINO?
    - Sortino only penalizes DOWNSIDE volatility
    - In trading, upside volatility is good!
    - Better metric for risk-adjusted performance than Sharpe
    
    MATH:
    Sortino = (R - Rf) / σ_d
    
    Where:
    - R = Expected return
    - Rf = Risk-free rate (often 0 for crypto)
    - σ_d = sqrt(E[min(R - Rf, 0)²]) = Downside deviation
    """
    
    def __init__(
        self,
        save_dir: str = './checkpoints',
        metric: str = 'sortino_ratio'
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.metric = metric
        self.best_metric = float('-inf')
        self.best_epoch = -1
    
    def save_if_best(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        config: Dict[str, Any]
    ) -> bool:
        """
        Save model if current metric is best.
        
        Returns:
            True if saved, False otherwise
        """
        current = metrics.get(self.metric, float('-inf'))
        
        if current > self.best_metric:
            self.best_metric = current
            self.best_epoch = epoch
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'config': config,
                'best_metric': self.best_metric
            }
            
            path = self.save_dir / f'best_model.pt'
            torch.save(checkpoint, path)
            
            # Also save metrics as JSON for easy viewing
            metrics_path = self.save_dir / 'best_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump({
                    'epoch': epoch,
                    'metrics': {k: float(v) for k, v in metrics.items()},
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            
            logger.info(
                f"✓ New best model saved! {self.metric}={current:.4f} (epoch {epoch})"
            )
            return True
        
        return False
    
    def load_best(self, model: nn.Module, device: str = 'cuda') -> Dict:
        """Load the best checkpoint."""
        path = self.save_dir / 'best_model.pt'
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint found at {path}")
        
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(
            f"Loaded best model from epoch {checkpoint['epoch']}, "
            f"{self.metric}={checkpoint['best_metric']:.4f}"
        )
        return checkpoint


# ==============================================================================
# MAIN TRAINER CLASS
# ==============================================================================
class OrionTrainer:
    """
    Complete training and backtesting pipeline for ORION.
    
    TRAINING FLOW:
    1. Initialize model, optimizer, scheduler
    2. Collect experience via environment interaction
    3. Train on batches from replay buffer
    4. Validate periodically, save best model
    5. Backtest on held-out test data
    
    VAST.AI OPTIMIZATIONS:
    - Mixed precision (FP16) for 2x batch size on RTX 4090
    - Gradient accumulation if batch doesn't fit
    - Pin memory for faster CPU→GPU transfer
    """
    
    def __init__(
        self,
        config: Dict[str, Any]
    ):
        """
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        logger.info(f"Using device: {self.device}")
        
        # Will be initialized in setup()
        self.model = None
        self.target_model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None  # For mixed precision
        self.saver = None
        self.replay_buffer = None
    
    def setup(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> None:
        """
        Initialize all training components.
        
        Args:
            train_data: Training set DataFrame
            val_data: Validation set DataFrame
            test_data: Test set (backtest) DataFrame
        """
        num_features = train_data.shape[1]
        lookback = self.config.get('lookback', 96)
        
        logger.info(f"Setting up training: {num_features} features, lookback={lookback}")
        
        # Create model
        model_config = {
            'hidden_dim': self.config.get('hidden_dim', 128),
            'num_heads': self.config.get('num_heads', 4),
            'num_lstm_layers': self.config.get('num_lstm_layers', 2),
            'num_quantiles': self.config.get('num_quantiles', 51),
            'num_actions': 4,
            'lookback': lookback,
            'dropout': self.config.get('dropout', 0.1)
        }
        
        self.model = create_model(num_features, model_config, str(self.device))
        self.target_model = create_model(num_features, model_config, str(self.device))
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {num_params:,}")
        
        # Optimizer (Ranger or AdamW)
        use_ranger = self.config.get('use_ranger', True)
        lr = self.config.get('lr', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-5)
        
        if use_ranger:
            self.optimizer = Ranger(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        
        # Scheduler: Cosine Annealing with Warm Restarts
        # T_0 = epochs per cycle, T_mult = cycle length multiplier
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.get('scheduler_t0', 10),
            T_mult=self.config.get('scheduler_t_mult', 2),
            eta_min=self.config.get('scheduler_eta_min', 1e-6)
        )
        
        # Mixed precision scaler (updated API for torch 2.x)
        self.scaler = GradScaler(device='cuda')
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=self.config.get('buffer_size', 100_000),
            state_shape=(lookback, num_features),
            device='cpu'  # Store on CPU to save GPU memory
        )
        
        # Model saver
        self.saver = ModelSaver(
            save_dir=self.config.get('checkpoint_dir', './checkpoints'),
            metric='sortino_ratio'
        )
        
        # Environments
        self.train_env = TradingEnvironment(
            train_data, lookback=lookback,
            transaction_cost=self.config.get('transaction_cost', 0.001)
        )
        self.val_env = TradingEnvironment(
            val_data, lookback=lookback,
            transaction_cost=self.config.get('transaction_cost', 0.001)
        )
        self.test_env = TradingEnvironment(
            test_data, lookback=lookback,
            transaction_cost=self.config.get('transaction_cost', 0.001)
        )
        
        logger.info("Training setup complete!")
    
    def _update_target_network(self, tau: float = 0.005) -> None:
        """
        Soft update of target network (Polyak averaging).
        
        MATH:
        θ' = τ * θ + (1 - τ) * θ'
        
        Where:
        - θ = online network weights
        - θ' = target network weights
        - τ = interpolation factor (small, like 0.005)
        
        This smoothly updates the target, stabilizing training.
        """
        for param, target_param in zip(
            self.model.parameters(),
            self.target_model.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )
    
    def collect_experience(
        self,
        num_steps: int,
        epsilon: float = 0.1,
        show_progress: bool = True
    ) -> Dict[str, float]:
        """
        Collect experience by interacting with environment.
        
        Args:
            num_steps: Number of environment steps
            epsilon: Exploration rate
            show_progress: Whether to show progress bar
            
        Returns:
            Statistics from collection
        """
        state = self.train_env.reset()
        total_reward = 0
        episode_rewards = []
        
        collection_start = time.time()
        for step in range(num_steps):
            # Progress updates every 10%
            if show_progress and step > 0 and step % max(1, num_steps // 10) == 0:
                pct = step / num_steps * 100
                elapsed = time.time() - collection_start
                eta = (elapsed / step) * (num_steps - step)
                logger.info(
                    f"  Collecting data: [{pct:3.0f}%] {step}/{num_steps} steps | "
                    f"ETA: {int(eta)}s | Buffer: {len(self.replay_buffer)}"
                )
            # Select action
            state_tensor = state.unsqueeze(0).to(self.device)
            action, _ = self.model.select_action(
                state_tensor,
                risk_level=self.config.get('risk_level', 'neutral'),
                epsilon=epsilon
            )
            action = action.item()
            
            # Step environment
            next_state, reward, done, info = self.train_env.step(action)
            total_reward += reward
            
            # Store transition
            self.replay_buffer.push(
                state, action, reward, next_state, done
            )
            
            state = next_state
            
            if done:
                episode_rewards.append(total_reward)
                total_reward = 0
                state = self.train_env.reset()
        
        return {
            'avg_episode_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'num_transitions': num_steps
        }
    
    def train_step(self, batch_size: int) -> float:
        """
        Single training step with mixed precision.
        
        Returns:
            Loss value
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size, str(self.device)
        )
        
        # Mixed precision forward pass
        with autocast(device_type='cuda'):
            loss = self.model.compute_loss(
                states, actions, rewards, next_states, dones,
                gamma=self.config.get('gamma', 0.99),
                target_net=self.target_model
            )
        
        # Backward pass with scaled gradients
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        
        # Gradient clipping (prevents exploding gradients)
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.get('max_grad_norm', 1.0)
        )
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
    
    def validate(self) -> Dict[str, float]:
        """
        Run validation episode and compute metrics.
        
        Returns:
            Performance metrics dictionary
        """
        self.model.eval()
        
        state = self.val_env.reset()
        done = False
        
        with torch.no_grad():
            while not done:
                state_tensor = state.unsqueeze(0).to(self.device)
                action, _ = self.model.select_action(
                    state_tensor,
                    risk_level=self.config.get('risk_level', 'neutral'),
                    epsilon=0.0  # Greedy
                )
                state, reward, done, info = self.val_env.step(action.item())
        
        metrics = self.val_env.get_performance_metrics()
        
        self.model.train()
        return metrics
    
    def _run_periodic_backtest(self, epoch: int) -> Dict[str, float]:
        """
        Run backtest on test data using CURRENT model (not saved checkpoint).
        
        This evaluates the current training state on held-out test data
        to track profitability during training.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Test performance metrics
        """
        backtest_start = time.time()
        
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"PERIODIC BACKTEST - Epoch {epoch+1}")
        logger.info("=" * 60)
        
        self.model.eval()
        
        state = self.test_env.reset()
        done = False
        actions_taken = []
        
        # Track starting balance for PnL
        initial_balance = self.test_env.initial_balance
        
        with torch.no_grad():
            while not done:
                state_tensor = state.unsqueeze(0).to(self.device)
                action, _ = self.model.select_action(
                    state_tensor,
                    risk_level=self.config.get('risk_level', 'neutral'),
                    epsilon=0.0
                )
                action_val = action.item()
                actions_taken.append(action_val)
                state, reward, done, info = self.test_env.step(action_val)
        
        backtest_time = time.time() - backtest_start
        
        metrics = self.test_env.get_performance_metrics()
        
        # Calculate PnL in dollars (assuming $10k starting capital)
        total_return = metrics.get('total_return', 0)
        final_balance = initial_balance * (1 + total_return)
        pnl_dollars = final_balance - initial_balance
        
        # Action distribution
        from collections import Counter
        action_counts = Counter(actions_taken)
        action_names = ['Long', 'Short', 'Hold', 'Close']
        
        # Trade analysis
        trades = self.test_env.trades
        if trades:
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] <= 0]
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        else:
            winning_trades = []
            losing_trades = []
            avg_win = 0
            avg_loss = 0
        
        # Output detailed results
        logger.info(f"Duration: {backtest_time:.1f}s | Steps: {len(actions_taken):,}")
        logger.info("-" * 40)
        logger.info("PERFORMANCE METRICS:")
        logger.info(f"  Return:        {total_return*100:+.2f}%")
        logger.info(f"  PnL:           ${pnl_dollars:+,.2f} (from ${initial_balance:,.0f})")
        logger.info(f"  Final Balance: ${final_balance:,.2f}")
        logger.info(f"  Sortino:       {metrics.get('sortino_ratio', 0):.4f}")
        logger.info(f"  Sharpe:        {metrics.get('sharpe_ratio', 0):.4f}")
        logger.info(f"  Max Drawdown:  {metrics.get('max_drawdown', 0)*100:.2f}%")
        logger.info("-" * 40)
        logger.info("TRADE STATISTICS:")
        logger.info(f"  Total Trades:  {len(trades)}")
        if trades:
            logger.info(f"  Winning:       {len(winning_trades)} ({len(winning_trades)/len(trades)*100:.1f}%)")
            logger.info(f"  Losing:        {len(losing_trades)} ({len(losing_trades)/len(trades)*100:.1f}%)")
        else:
            logger.info(f"  Winning:       0")
            logger.info(f"  Losing:        0")
        logger.info(f"  Avg Win:       {avg_win*100:+.2f}%")
        logger.info(f"  Avg Loss:      {avg_loss*100:+.2f}%")
        logger.info(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        logger.info("-" * 40)
        logger.info("ACTION DISTRIBUTION:")
        for i, name in enumerate(action_names):
            count = action_counts.get(i, 0)
            pct = count / len(actions_taken) * 100 if actions_taken else 0
            bar = '█' * int(pct / 5)  # Simple bar chart
            logger.info(f"  {name:6s}: {count:5d} ({pct:5.1f}%) {bar}")
        logger.info("=" * 60)
        
        self.model.train()
        return metrics
    
    def _save_best_profitable_model(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Save the most profitable model based on test return.
        
        This is separate from validation-based checkpointing.
        Saves to 'best_profitable_model.pt'.
        
        Args:
            epoch: Current epoch
            metrics: Test metrics
        """
        save_path = Path(self.saver.save_dir) / 'best_profitable_model.pt'
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'test_metrics': metrics,
            'config': self.config,
            'best_return': self.best_test_return
        }
        
        torch.save(checkpoint, save_path)
        
        logger.info(f"✓ NEW BEST PROFITABLE MODEL SAVED!")
        logger.info(f"  Epoch: {epoch+1}, Return: {metrics.get('total_return', 0)*100:.2f}%")
        logger.info(f"  Path: {save_path}")
        
        # Also save metrics as JSON
        metrics_path = Path(self.saver.save_dir) / 'best_profitable_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump({
                'epoch': epoch,
                'metrics': {k: float(v) for k, v in metrics.items()},
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
    
    def train(self) -> Dict[str, List[float]]:
        """
        Main training loop with detailed progress logging.
        
        Returns:
            Training history dictionary
        """
        num_epochs = self.config.get('num_epochs', 100)
        steps_per_epoch = self.config.get('steps_per_epoch', 1000)
        batch_size = self.config.get('batch_size', 256)
        updates_per_step = self.config.get('updates_per_step', 4)
        initial_epsilon = self.config.get('initial_epsilon', 1.0)
        final_epsilon = self.config.get('final_epsilon', 0.01)
        epsilon_decay = self.config.get('epsilon_decay', 0.995)
        target_update_freq = self.config.get('target_update_freq', 100)
        
        history = {
            'loss': [],
            'val_sortino': [],
            'val_sharpe': [],
            'epsilon': [],
            'test_return': [],  # Periodic backtest results
            'test_sortino': []
        }
        
        # Periodic backtest config
        backtest_freq = self.config.get('backtest_freq', 10)  # Every N epochs
        self.best_test_return = float('-inf')
        self.best_test_epoch = -1
        
        epsilon = initial_epsilon
        total_steps = 0
        training_start = time.time()
        
        # Calculate total updates for progress tracking
        total_updates = num_epochs * steps_per_epoch * updates_per_step
        
        logger.info("=" * 60)
        logger.info("TRAINING CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Steps/epoch: {steps_per_epoch}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Updates/step: {updates_per_step}")
        logger.info(f"Total gradient updates: {total_updates:,}")
        logger.info(f"Replay buffer size: {len(self.replay_buffer)}/{self.replay_buffer.capacity}")
        logger.info(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
        logger.info(f"Device: {self.device}")
        
        # GPU info with memory usage
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_mem_used = torch.cuda.memory_allocated() / 1024**3
            gpu_mem_pct = gpu_mem_used / gpu_mem_total * 100
            logger.info(f"GPU: {gpu_name} ({gpu_mem_total:.1f} GB)")
            logger.info(f"GPU Memory: {gpu_mem_used:.1f}/{gpu_mem_total:.1f} GB ({gpu_mem_pct:.1f}%) - Model loaded")
        
        logger.info("=" * 60)
        logger.info("STARTING TRAINING")
        logger.info("=" * 60)
        logger.info("")
        logger.info("⏳ Epoch 1 starting... (this may take 2-3 minutes)")
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            epoch_losses = []
            
            # ================================================================
            # PHASE 1: Experience Collection
            # ================================================================
            logger.info(f"EPOCH {epoch+1}/{num_epochs} - Collecting {steps_per_epoch} steps...")
            self.model.train()
            collect_start = time.time()
            collection_stats = self.collect_experience(
                steps_per_epoch,
                epsilon=epsilon,
                show_progress=True
            )
            collect_time = time.time() - collect_start
            
            # ================================================================
            # PHASE 2: Training Updates with Progress
            # ================================================================
            update_start = time.time()
            num_updates = steps_per_epoch * updates_per_step
            
            # Skip training if buffer doesn't have enough samples
            min_samples_for_training = batch_size
            if len(self.replay_buffer) < min_samples_for_training:
                logger.info(f"Buffer has {len(self.replay_buffer)} samples, need {min_samples_for_training}. Skipping training this epoch.")
                num_updates = 0
            
            for update_idx in range(num_updates):
                loss = self.train_step(batch_size)
                if loss > 0:  # Only track non-zero losses
                    epoch_losses.append(loss)
                
                total_steps += 1
                
                # Update target network
                if total_steps % target_update_freq == 0:
                    self._update_target_network()
                
                # Progress logging every 25%
                if (update_idx + 1) % (num_updates // 4) == 0:
                    progress_pct = (update_idx + 1) / num_updates * 100
                    avg_loss_so_far = np.mean(epoch_losses) if epoch_losses else 0
                    
                    # GPU memory usage
                    if torch.cuda.is_available():
                        gpu_mem_used = torch.cuda.memory_allocated() / 1024**3
                        gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        gpu_pct = gpu_mem_used / gpu_mem_total * 100
                        logger.info(
                            f"  [{progress_pct:3.0f}%] Updates: {update_idx+1}/{num_updates} | "
                            f"Loss: {avg_loss_so_far:.4f} | "
                            f"GPU Mem: {gpu_mem_used:.1f}/{gpu_mem_total:.1f}GB ({gpu_pct:.0f}%)"
                        )
            
            update_time = time.time() - update_start
            
            # Decay epsilon
            epsilon = max(final_epsilon, epsilon * epsilon_decay)
            
            # Scheduler step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # ================================================================
            # PHASE 3: Validation (skip first 10 epochs for speed)
            # ================================================================
            val_start = time.time()
            if epoch < 10:
                # Skip validation in early epochs (model not trained yet)
                val_metrics = {'sortino_ratio': 0.0, 'sharpe_ratio': 0.0}
                val_time = 0.0
            else:
                val_metrics = self.validate()
                val_time = time.time() - val_start
            
            # Record history
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0
            history['loss'].append(avg_loss)
            history['val_sortino'].append(val_metrics.get('sortino_ratio', 0))
            history['val_sharpe'].append(val_metrics.get('sharpe_ratio', 0))
            history['epsilon'].append(epsilon)
            
            # Save if best (validation-based)
            self.saver.save_if_best(
                self.model,
                self.optimizer,
                epoch,
                val_metrics,
                self.config
            )
            
            # ================================================================
            # PERIODIC BACKTEST (every N epochs) - saves most profitable model
            # ================================================================
            if backtest_freq > 0 and (epoch + 1) % backtest_freq == 0:
                test_metrics = self._run_periodic_backtest(epoch)
                history['test_return'].append(test_metrics.get('total_return', 0))
                history['test_sortino'].append(test_metrics.get('sortino_ratio', 0))
                
                # Save if most profitable
                current_return = test_metrics.get('total_return', float('-inf'))
                if current_return > self.best_test_return:
                    self.best_test_return = current_return
                    self.best_test_epoch = epoch
                    self._save_best_profitable_model(epoch, test_metrics)
            else:
                # Append NaN for non-backtest epochs (to keep aligned)
                history['test_return'].append(float('nan'))
                history['test_sortino'].append(float('nan'))
            
            # ================================================================
            # EPOCH SUMMARY with timing breakdown
            # ================================================================
            epoch_time = time.time() - epoch_start
            elapsed_total = time.time() - training_start
            epochs_remaining = num_epochs - epoch - 1
            eta_seconds = (elapsed_total / (epoch + 1)) * epochs_remaining if epoch > 0 else 0
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
            
            logger.info("-" * 60)
            logger.info(
                f"EPOCH {epoch+1}/{num_epochs} COMPLETE | "
                f"Time: {epoch_time:.1f}s | "
                f"ETA: {eta_str}"
            )
            logger.info(
                f"  Loss: {avg_loss:.4f} | "
                f"Val Sortino: {val_metrics.get('sortino_ratio', 0):.4f} | "
                f"Val Sharpe: {val_metrics.get('sharpe_ratio', 0):.4f}"
            )
            logger.info(
                f"  ε: {epsilon:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Buffer: {len(self.replay_buffer):,}"
            )
            logger.info(
                f"  Timing: Collect={collect_time:.1f}s, Train={update_time:.1f}s, Val={val_time:.1f}s"
            )
            logger.info("-" * 60)
        
        # Final summary
        total_time = time.time() - training_start
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
        logger.info(f"Best validation Sortino: {max(history['val_sortino']):.4f}")
        if self.best_test_epoch >= 0:
            logger.info(f"Best test return: {self.best_test_return*100:.2f}% (epoch {self.best_test_epoch+1})")
        
        return history
    
    def backtest(self) -> Dict[str, float]:
        """
        Final backtest on held-out test data with comprehensive output.
        
        Loads the best model and runs a full episode on test data.
        
        Returns:
            Test performance metrics
        """
        backtest_start = time.time()
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("FINAL BACKTEST ON HELD-OUT TEST DATA")
        logger.info("=" * 60)
        
        # Load best model
        try:
            self.saver.load_best(self.model, str(self.device))
            logger.info("✓ Loaded best validation model")
        except FileNotFoundError:
            logger.warning("No saved model found, using current model")
        
        self.model.eval()
        
        state = self.test_env.reset()
        done = False
        actions_taken = []
        initial_balance = self.test_env.initial_balance
        
        with torch.no_grad():
            while not done:
                state_tensor = state.unsqueeze(0).to(self.device)
                action, q_value = self.model.select_action(
                    state_tensor,
                    risk_level=self.config.get('risk_level', 'neutral'),
                    epsilon=0.0
                )
                action = action.item()
                actions_taken.append(action)
                
                state, reward, done, info = self.test_env.step(action)
        
        backtest_time = time.time() - backtest_start
        
        # Compute metrics
        metrics = self.test_env.get_performance_metrics()
        
        # Calculate PnL
        total_return = metrics.get('total_return', 0)
        final_balance = initial_balance * (1 + total_return)
        pnl_dollars = final_balance - initial_balance
        
        # Action distribution
        from collections import Counter
        action_counts = Counter(actions_taken)
        action_names = ['Long', 'Short', 'Hold', 'Close']
        
        # Trade analysis
        trades = self.test_env.trades
        if trades:
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] <= 0]
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            best_trade = max(t['pnl'] for t in trades)
            worst_trade = min(t['pnl'] for t in trades)
        else:
            winning_trades = []
            losing_trades = []
            avg_win = avg_loss = best_trade = worst_trade = 0
        
        # Price range info
        prices = self.test_env.prices
        price_start = prices[self.test_env.lookback] if len(prices) > self.test_env.lookback else prices[0]
        price_end = prices[-1]
        buy_hold_return = (price_end - price_start) / price_start
        
        logger.info(f"Duration: {backtest_time:.1f}s | Steps: {len(actions_taken):,}")
        logger.info("")
        logger.info("=" * 40)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 40)
        logger.info(f"  Strategy Return:   {total_return*100:+.2f}%")
        logger.info(f"  Buy & Hold Return: {buy_hold_return*100:+.2f}%")
        logger.info(f"  Alpha:             {(total_return - buy_hold_return)*100:+.2f}%")
        logger.info("")
        logger.info(f"  Starting Capital:  ${initial_balance:,.0f}")
        logger.info(f"  Final Balance:     ${final_balance:,.2f}")
        logger.info(f"  Net PnL:           ${pnl_dollars:+,.2f}")
        logger.info("")
        logger.info("=" * 40)
        logger.info("RISK METRICS")
        logger.info("=" * 40)
        logger.info(f"  Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):.4f}")
        logger.info(f"  Sortino Ratio:     {metrics.get('sortino_ratio', 0):.4f}")
        logger.info(f"  Max Drawdown:      {metrics.get('max_drawdown', 0)*100:.2f}%")
        logger.info("")
        logger.info("=" * 40)
        logger.info("TRADE ANALYSIS")
        logger.info("=" * 40)
        logger.info(f"  Total Trades:      {len(trades)}")
        logger.info(f"  Win Rate:          {metrics.get('win_rate', 0)*100:.1f}%")
        logger.info(f"  Profit Factor:     {metrics.get('profit_factor', 0):.2f}")
        if trades:
            logger.info(f"  Winning Trades:    {len(winning_trades)}")
            logger.info(f"  Losing Trades:     {len(losing_trades)}")
            logger.info(f"  Avg Win:           {avg_win*100:+.2f}%")
            logger.info(f"  Avg Loss:          {avg_loss*100:+.2f}%")
            logger.info(f"  Best Trade:        {best_trade*100:+.2f}%")
            logger.info(f"  Worst Trade:       {worst_trade*100:+.2f}%")
        logger.info("")
        logger.info("=" * 40)
        logger.info("ACTION DISTRIBUTION")
        logger.info("=" * 40)
        for i, name in enumerate(action_names):
            count = action_counts.get(i, 0)
            pct = count / len(actions_taken) * 100 if actions_taken else 0
            bar = '█' * int(pct / 5)
            logger.info(f"  {name:6s}: {count:6d} ({pct:5.1f}%) {bar}")
        logger.info("")
        logger.info("=" * 60)
        logger.info("BACKTEST COMPLETE")
        logger.info("=" * 60)
        
        return metrics


# ==============================================================================
# DATA VALIDATION
# ==============================================================================
def validate_data(df: pd.DataFrame, stage: str = "unknown") -> bool:
    """
    Validate data before training to catch issues early.
    
    Checks for:
    - NaN/Inf values
    - Minimum data length
    - Feature count
    - Value ranges (detects scaling issues)
    
    Args:
        df: DataFrame to validate
        stage: Name of the stage (for logging)
        
    Returns:
        True if valid, raises ValueError if not
    """
    logger.info(f"\n{'='*40}")
    logger.info(f"DATA VALIDATION - {stage}")
    logger.info(f"{'='*40}")
    
    issues = []
    
    # Check shape
    logger.info(f"Shape: {df.shape} (rows × features)")
    
    if len(df) < 1000:
        issues.append(f"Insufficient data: {len(df)} rows (need >= 1000)")
    
    # Check for NaN
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        nan_cols = df.columns[df.isna().any()].tolist()
        issues.append(f"Found {nan_count} NaN values in columns: {nan_cols[:5]}...")
    else:
        logger.info("✓ No NaN values")
    
    # Check for Inf
    inf_count = np.isinf(df.values).sum()
    if inf_count > 0:
        issues.append(f"Found {inf_count} Inf values")
    else:
        logger.info("✓ No Inf values")
    
    # Check value ranges (after normalization, values should be reasonable)
    max_val = df.values.max()
    min_val = df.values.min()
    logger.info(f"Value range: [{min_val:.4f}, {max_val:.4f}]")
    
    if abs(max_val) > 1000 or abs(min_val) > 1000:
        logger.warning(f"⚠ Large values detected. Consider checking normalization.")
        # Diagnose which columns have the largest magnitudes (helpful to find volume/cvd)
        abs_max_per_col = df.abs().max()
        large_cols = abs_max_per_col[abs_max_per_col > 1000].sort_values(ascending=False)
        if len(large_cols) > 0:
            logger.warning("Top columns by absolute magnitude:")
            for col, val in large_cols.head(10).items():
                logger.warning(f"  {col}: max_abs={val:.4f}, min={df[col].min():.4f}, max={df[col].max():.4f}")
            # Suggest common fixes when large values are found
            logger.warning("Consider applying log1p to strictly positive features like volume, or Winsorize/clip outliers.")
        # Also report which column contains the overall min/max values
        try:
            min_col = df.min().idxmin()
            max_col = df.max().idxmax()
            logger.info(f"Column with min value: {min_col} ({df[min_col].min():.4f})")
            logger.info(f"Column with max value: {max_col} ({df[max_col].max():.4f})")
        except Exception:
            # If for some reason index operations fail, we skip
            pass
    
    # Check for constant columns (zero variance)
    std = df.std()
    zero_var_cols = std[std == 0].index.tolist()
    if zero_var_cols:
        logger.warning(f"⚠ Zero-variance columns (will be ignored by model): {zero_var_cols[:5]}")
    
    # Feature statistics
    logger.info(f"Feature statistics (sample):")
    sample_cols = df.columns[:5].tolist()
    for col in sample_cols:
        logger.info(f"  {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")
    
    # Extra diagnostics: show top-3 columns by std (helps find very noisy features)
    std_sorted = df.std().sort_values(ascending=False)
    logger.info("Top 3 features by std deviation:")
    for col, s in std_sorted.head(3).items():
        logger.info(f"  {col}: std={s:.4f} (min={df[col].min():.4f}, max={df[col].max():.4f})")
    
    # Report issues
    if issues:
        for issue in issues:
            logger.error(f"✗ {issue}")
        raise ValueError(f"Data validation failed at stage '{stage}': {issues}")
    
    logger.info(f"✓ Data validation PASSED for stage: {stage}")
    return True


# ==============================================================================
# DATA PREPROCESSING PIPELINE
# ==============================================================================
def get_preprocessed_cache_path(cache_dir: str, years: int) -> Path:
    """Get the path for cached preprocessed data."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path / f"preprocessed_{years}y.parquet"


def save_preprocessed_data(df: pd.DataFrame, cache_path: Path, d_values: dict = None) -> None:
    """
    Save preprocessed data to disk.
    
    Args:
        df: Preprocessed DataFrame
        cache_path: Path to save the data
        d_values: FFD d values for each column (for logging on reload)
    """
    df.to_parquet(cache_path, engine='pyarrow', compression='snappy')
    
    # Save metadata (FFD d values, timestamp, etc.)
    meta_path = cache_path.with_suffix('.json')
    metadata = {
        'created': datetime.now().isoformat(),
        'shape': list(df.shape),
        'columns': list(df.columns),
        'd_values': d_values or {},
        'index_start': str(df.index[0]) if len(df) > 0 else None,
        'index_end': str(df.index[-1]) if len(df) > 0 else None,
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✓ Saved preprocessed data to {cache_path}")
    logger.info(f"  Shape: {df.shape}, Size: {cache_path.stat().st_size / 1024 / 1024:.2f} MB")


def load_preprocessed_data(cache_path: Path) -> Optional[pd.DataFrame]:
    """
    Load preprocessed data from cache if it exists and is valid.
    
    Args:
        cache_path: Path to the cached data
        
    Returns:
        DataFrame if cache exists and is valid, None otherwise
    """
    if not cache_path.exists():
        return None
    
    meta_path = cache_path.with_suffix('.json')
    if not meta_path.exists():
        logger.warning(f"Cache metadata missing: {meta_path}")
        return None
    
    try:
        # Load metadata
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        # Check cache age (invalidate if older than 24 hours)
        created = datetime.fromisoformat(metadata['created'])
        age_hours = (datetime.now() - created).total_seconds() / 3600
        
        if age_hours > 24:
            logger.info(f"Cache is {age_hours:.1f} hours old, will refresh")
            return None
        
        # Load data
        df = pd.read_parquet(cache_path)
        
        logger.info(f"✓ Loaded preprocessed data from cache")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Created: {metadata['created']}")
        logger.info(f"  Date range: {metadata.get('index_start', 'N/A')} to {metadata.get('index_end', 'N/A')}")
        
        # Log FFD d values if available
        if metadata.get('d_values'):
            logger.info("  FFD d values used:")
            for col, d in list(metadata['d_values'].items())[:5]:
                logger.info(f"    {col}: d={d:.2f}")
            if len(metadata['d_values']) > 5:
                logger.info(f"    ... and {len(metadata['d_values']) - 5} more columns")
        
        return df
        
    except Exception as e:
        logger.warning(f"Failed to load cache: {e}")
        return None


def preprocess_data(
    df: pd.DataFrame,
    aligner: MultiTimeframeAligner,
    cache_dir: str = './data_cache',
    years: int = 5,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Apply FFD and normalization to prepared data.
    
    Caching behavior:
    - Saves preprocessed data to disk after processing
    - Loads from cache if available and < 24 hours old
    - Cache is invalidated if 'years' parameter changes
    
    Args:
        df: Aligned DataFrame with all features
        aligner: Fitted aligner (for normalization)
        cache_dir: Directory for cache files
        years: Years of data (used in cache filename)
        use_cache: Whether to use caching
        
    Returns:
        Preprocessed DataFrame
    """
    # Check cache first
    if use_cache:
        cache_path = get_preprocessed_cache_path(cache_dir, years)
        cached_df = load_preprocessed_data(cache_path)
        if cached_df is not None:
            return cached_df
    
    logger.info("Applying Fractional Differencing...")
    
    # === CRITICAL: Preserve raw close prices for trading PnL calculation ===
    # FFD transforms prices to stationary values, but we need actual prices for trading
    raw_close_col = [c for c in df.columns if 'close_5m' in c.lower()]
    if raw_close_col:
        # Store raw close prices (will be normalized but NOT FFD-transformed)
        df['_raw_close_5m'] = df[raw_close_col[0]].copy()
        logger.info(f"Preserved raw close prices in '_raw_close_5m' for PnL calculation")
    
    # Identify OHLCV columns (need FFD)
    ohlcv_cols = [c for c in df.columns if any(
        x in c.lower() for x in ['open', 'high', 'low', 'close', 'volume']
    ) and not any(y in c.lower() for y in ['ema', 'rsi', 'atr', 'bb', 'adx', 'vwap', 'cvd', 'ichimoku', 'tsi', '_raw_'])]
    
    # Apply FFD to OHLCV columns
    ffd = FastFractionalDiff(d_step=0.05, significance=0.05)
    d_values = {}
    
    # === Apply log1p to volume columns first (handles extreme values) ===
    volume_cols = [c for c in ohlcv_cols if 'volume' in c.lower()]
    for vol_col in volume_cols:
        # Volume is always positive, log1p handles zeros gracefully
        df[vol_col] = np.log1p(df[vol_col].clip(lower=0))
    if volume_cols:
        logger.info(f"Applied log1p to {len(volume_cols)} volume columns")
    
    if ohlcv_cols:
        df_ohlcv = df[ohlcv_cols]
        df_ffd = ffd.fit_transform(df_ohlcv)
        
        # Replace original columns with FFD versions
        for col in ohlcv_cols:
            df[col] = df_ffd[col]
        
        d_values = ffd.d_values.copy()
        
        logger.info(f"FFD applied to {len(ohlcv_cols)} OHLCV columns")
        logger.info("Optimal d values:")
        for col, d in ffd.d_values.items():
            logger.info(f"  {col}: d={d:.2f}")
    
    # Drop NaN rows (from FFD warm-up)
    initial_len = len(df)
    df = df.dropna()
    logger.info(f"Dropped {initial_len - len(df)} rows (FFD warm-up)")
    
    # === CRITICAL: Save truly raw prices BEFORE normalization ===
    # The _raw_close_5m column was saved before FFD, but normalization will change it
    # We need actual dollar prices for PnL calculation
    raw_prices_backup = None
    if '_raw_close_5m' in df.columns:
        raw_prices_backup = df['_raw_close_5m'].copy()
    
    # Normalize (this will scale _raw_close_5m too, but we'll restore it)
    logger.info("Applying RobustScaler normalization...")
    df = aligner.normalize(df, fit=True)
    
    # === Restore truly raw prices (not normalized, not FFD) for trading ===
    if raw_prices_backup is not None:
        df['_raw_close_5m'] = raw_prices_backup
        logger.info("Restored raw close prices (bypassed normalization for PnL accuracy)")
    
    # Save to cache
    if use_cache:
        cache_path = get_preprocessed_cache_path(cache_dir, years)
        save_preprocessed_data(df, cache_path, d_values)
    
    return df


def train_test_split_ts(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-series aware train/val/test split.
    
    IMPORTANT: No shuffling! We maintain temporal order.
    
    Args:
        df: Full dataset
        train_ratio: Training set fraction
        val_ratio: Validation set fraction
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    return train_df, val_df, test_df


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
def get_optimal_batch_size() -> int:
    """
    Determine optimal batch size based on available GPU memory.
    Targets 80-90% GPU memory utilization.
    """
    if not torch.cuda.is_available():
        return 256
    
    # Get total GPU memory
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    # Rough estimate: batch_size scales with available memory
    # RTX 4090 (24GB) -> 512, RTX 3090 (24GB) -> 512, RTX 3080 (10GB) -> 256
    if total_mem_gb >= 20:
        return 512
    elif total_mem_gb >= 12:
        return 384
    else:
        return 256


def main():
    """Main training pipeline."""
    
    # Auto-detect optimal batch size
    optimal_batch = get_optimal_batch_size()
    
    # Configuration
    config = {
        # Data
        'years_of_history': 5,
        'cache_dir': './data_cache',
        
        # Model - Reduced for memory efficiency
        'hidden_dim': 128,      # Reduced from 256
        'num_heads': 4,         # Reduced from 8
        'num_lstm_layers': 2,
        'num_quantiles': 25,    # Reduced from 51
        'lookback': 64,         # 5.3 hours at 5m (reduced from 96)
        'dropout': 0.1,
        
        # Training - Optimized for speed (10GB VRAM usage leaves room)
        'num_epochs': 100,
        'batch_size': optimal_batch,  # Auto-detected based on GPU memory
        'steps_per_epoch': 1500, # Ensures buffer > batch_size for training
        'updates_per_step': 8,  # More GPU work per collected step
        'lr': 3e-4,
        'weight_decay': 1e-5,
        'gamma': 0.99,
        'max_grad_norm': 1.0,
        'use_ranger': True,
        
        # Scheduler
        'scheduler_t0': 10,
        'scheduler_t_mult': 2,
        'scheduler_eta_min': 1e-6,
        
        # Exploration
        'initial_epsilon': 1.0,
        'final_epsilon': 0.01,
        'epsilon_decay': 0.995,
        
        # Replay buffer - larger for diversity
        'buffer_size': 200_000,
        
        # Target network
        'target_update_freq': 100,
        
        # Trading
        'transaction_cost': 0.001,
        'risk_level': 'neutral',
        
        # Periodic backtest
        'backtest_freq': 20,  # Run backtest every N epochs (reduced for speed)
        
        # Checkpoints
        'checkpoint_dir': './checkpoints'
    }
    
    logger.info("=" * 60)
    logger.info("O.R.I.O.N. - Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"Config: {json.dumps(config, indent=2)}")
    
    # Load and prepare data
    logger.info("\nLoading data...")
    df, aligner = load_and_prepare_data(
        years=config['years_of_history'],
        cache_dir=config['cache_dir'],
        use_cache=True
    )
    
    # Preprocess (FFD + normalization) - uses caching
    logger.info("\nPreprocessing data...")
    df = preprocess_data(
        df, 
        aligner,
        cache_dir=config['cache_dir'],
        years=config['years_of_history'],
        use_cache=True
    )
    
    # Validate data after preprocessing
    validate_data(df, stage="After Preprocessing")
    
    # Split data
    logger.info("\nSplitting data...")
    train_df, val_df, test_df = train_test_split_ts(df)
    
    # Validate splits
    validate_data(train_df, stage="Training Set")
    validate_data(val_df, stage="Validation Set")
    validate_data(test_df, stage="Test Set")
    
    # Initialize trainer
    trainer = OrionTrainer(config)
    trainer.setup(train_df, val_df, test_df)
    
    # Train
    logger.info("\nStarting training...")
    history = trainer.train()
    
    # Backtest
    logger.info("\nRunning backtest...")
    test_metrics = trainer.backtest()
    
    # Save final results
    results = {
        'config': config,
        'history': history,
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = Path(config['checkpoint_dir']) / 'final_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {results_path}")
    logger.info("Training complete!")
    
    return results


if __name__ == "__main__":
    main()
