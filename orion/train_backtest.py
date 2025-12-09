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
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Local imports
from orion.data.loader import OrionDataLoader, MultiTimeframeAligner, load_and_prepare_data
from orion.math.fracdiff import FastFractionalDiff
from orion.models.hybrid import TFT_QR_DQN, create_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger("ORION.Train")


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
        # Find the 5m close column (base timeframe)
        close_col = [c for c in data.columns if 'close_5m' in c.lower()]
        if close_col:
            self.prices = data[close_col[0]].values
        else:
            # Fallback: use first 'close' column
            close_cols = [c for c in data.columns if 'close' in c.lower()]
            self.prices = data[close_cols[0]].values if close_cols else data.iloc[:, 3].values
        
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
        
        # Mixed precision scaler
        self.scaler = GradScaler()
        
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
        epsilon: float = 0.1
    ) -> Dict[str, float]:
        """
        Collect experience by interacting with environment.
        
        Args:
            num_steps: Number of environment steps
            epsilon: Exploration rate
            
        Returns:
            Statistics from collection
        """
        state = self.train_env.reset()
        total_reward = 0
        episode_rewards = []
        
        for step in range(num_steps):
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
        with autocast():
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
    
    def train(self) -> Dict[str, List[float]]:
        """
        Main training loop.
        
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
            'epsilon': []
        }
        
        epsilon = initial_epsilon
        total_steps = 0
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Batch size: {batch_size}, Steps/epoch: {steps_per_epoch}")
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            epoch_losses = []
            
            # Collect experience
            self.model.train()
            collection_stats = self.collect_experience(
                steps_per_epoch,
                epsilon=epsilon
            )
            
            # Training updates
            for _ in range(steps_per_epoch * updates_per_step):
                loss = self.train_step(batch_size)
                epoch_losses.append(loss)
                
                total_steps += 1
                
                # Update target network
                if total_steps % target_update_freq == 0:
                    self._update_target_network()
            
            # Decay epsilon
            epsilon = max(final_epsilon, epsilon * epsilon_decay)
            
            # Scheduler step
            self.scheduler.step()
            
            # Validation
            val_metrics = self.validate()
            
            # Record history
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0
            history['loss'].append(avg_loss)
            history['val_sortino'].append(val_metrics.get('sortino_ratio', 0))
            history['val_sharpe'].append(val_metrics.get('sharpe_ratio', 0))
            history['epsilon'].append(epsilon)
            
            # Save if best
            self.saver.save_if_best(
                self.model,
                self.optimizer,
                epoch,
                val_metrics,
                self.config
            )
            
            # Logging
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Val Sortino: {val_metrics.get('sortino_ratio', 0):.4f} | "
                f"Val Sharpe: {val_metrics.get('sharpe_ratio', 0):.4f} | "
                f"ε: {epsilon:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )
        
        return history
    
    def backtest(self) -> Dict[str, float]:
        """
        Final backtest on held-out test data.
        
        Loads the best model and runs a full episode on test data.
        
        Returns:
            Test performance metrics
        """
        logger.info("=" * 60)
        logger.info("BACKTEST ON HELD-OUT TEST DATA")
        logger.info("=" * 60)
        
        # Load best model
        try:
            self.saver.load_best(self.model, str(self.device))
        except FileNotFoundError:
            logger.warning("No saved model found, using current model")
        
        self.model.eval()
        
        state = self.test_env.reset()
        done = False
        actions_taken = []
        
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
        
        # Compute metrics
        metrics = self.test_env.get_performance_metrics()
        
        # Action distribution
        from collections import Counter
        action_counts = Counter(actions_taken)
        action_names = ['Long', 'Short', 'Hold', 'Close']
        
        logger.info("\n" + "=" * 40)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 40)
        logger.info(f"Total Return: {metrics.get('total_return', 0)*100:.2f}%")
        logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
        logger.info(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.4f}")
        logger.info(f"Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
        logger.info(f"Win Rate: {metrics.get('win_rate', 0)*100:.2f}%")
        logger.info(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        logger.info(f"Num Trades: {metrics.get('num_trades', 0)}")
        
        logger.info("\nAction Distribution:")
        for i, name in enumerate(action_names):
            count = action_counts.get(i, 0)
            pct = count / len(actions_taken) * 100
            logger.info(f"  {name}: {count} ({pct:.1f}%)")
        
        return metrics


# ==============================================================================
# DATA PREPROCESSING PIPELINE
# ==============================================================================
def preprocess_data(
    df: pd.DataFrame,
    aligner: MultiTimeframeAligner
) -> pd.DataFrame:
    """
    Apply FFD and normalization to prepared data.
    
    Args:
        df: Aligned DataFrame with all features
        aligner: Fitted aligner (for normalization)
        
    Returns:
        Preprocessed DataFrame
    """
    logger.info("Applying Fractional Differencing...")
    
    # Identify OHLCV columns (need FFD)
    ohlcv_cols = [c for c in df.columns if any(
        x in c.lower() for x in ['open', 'high', 'low', 'close', 'volume']
    ) and not any(y in c.lower() for y in ['ema', 'rsi', 'atr', 'bb', 'adx', 'vwap', 'cvd', 'ichimoku', 'tsi'])]
    
    # Apply FFD to OHLCV columns
    ffd = FastFractionalDiff(d_step=0.05, significance=0.05)
    
    if ohlcv_cols:
        df_ohlcv = df[ohlcv_cols]
        df_ffd = ffd.fit_transform(df_ohlcv)
        
        # Replace original columns with FFD versions
        for col in ohlcv_cols:
            df[col] = df_ffd[col]
        
        logger.info(f"FFD applied to {len(ohlcv_cols)} OHLCV columns")
        logger.info("Optimal d values:")
        for col, d in ffd.d_values.items():
            logger.info(f"  {col}: d={d:.2f}")
    
    # Drop NaN rows (from FFD warm-up)
    initial_len = len(df)
    df = df.dropna()
    logger.info(f"Dropped {initial_len - len(df)} rows (FFD warm-up)")
    
    # Normalize
    logger.info("Applying RobustScaler normalization...")
    df = aligner.normalize(df, fit=True)
    
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
def main():
    """Main training pipeline."""
    
    # Configuration
    config = {
        # Data
        'years_of_history': 5,
        'cache_dir': './data_cache',
        
        # Model
        'hidden_dim': 128,
        'num_heads': 4,
        'num_lstm_layers': 2,
        'num_quantiles': 51,
        'lookback': 96,  # 8 hours at 5m
        'dropout': 0.1,
        
        # Training
        'num_epochs': 100,
        'batch_size': 256,
        'steps_per_epoch': 1000,
        'updates_per_step': 4,
        'lr': 1e-4,
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
        
        # Replay buffer
        'buffer_size': 100_000,
        
        # Target network
        'target_update_freq': 100,
        
        # Trading
        'transaction_cost': 0.001,
        'risk_level': 'neutral',
        
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
    
    # Preprocess (FFD + normalization)
    logger.info("\nPreprocessing data...")
    df = preprocess_data(df, aligner)
    
    # Split data
    logger.info("\nSplitting data...")
    train_df, val_df, test_df = train_test_split_ts(df)
    
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
