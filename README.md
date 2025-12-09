# O.R.I.O.N. 
## Omniscient Risk & Investment Optimization Network

Production-grade GPU-accelerated Deep Reinforcement Learning trading system for Bitcoin (BTC/USDT) using a hybrid **Temporal Fusion Transformer (TFT)** and **Quantile Regression DQN (QR-DQN)** architecture.

---

## Architecture Overview

```
Multi-Timeframe Input (5m, 15m, 1h, 4h, 1d)
           ↓
    [Variable Selection Network]
    Learns dynamic feature importance
           ↓
    [Gated Residual Networks]
    Filters noise via gating
           ↓
    [Multi-Head Attention]
    Long-range temporal dependencies
           ↓
    [Context Embedding e_t]
           ↓
    [Quantile Regression DQN]
    Distributional RL for risk-aware actions
           ↓
    Action: Long | Short | Hold | Close
```

## Key Features

### Data Pipeline (`orion/data/loader.py`)
- **Fail-over loader**: Binance → BinanceUS → CSV fallback
- **Multi-timeframe alignment**: Master 5m index with forward-fill
- **Technical indicators**: EMA Ribbon, RSI, TSI, ADX, Bollinger Bands, VWAP, CVD

### Fractional Differencing (`orion/math/fracdiff.py`)
- **FFD (Fast Fractional Differencing)**: Stationarity with memory preservation
- **Automatic d optimization**: ADF testing to find minimum d for stationarity
- **Numba-accelerated**: JIT-compiled weight computation

### Hybrid Model (`orion/models/hybrid.py`)
- **TFT Encoder**: Variable Selection + LSTM + Multi-Head Attention
- **QR-DQN Agent**: 51-quantile distributional RL
- **Risk-aware actions**: Neutral, averse, or seeking risk profiles

### Training Pipeline (`orion/train_backtest.py`)
- **Mixed Precision**: FP16 for 2x batch size on RTX 4090
- **Ranger Optimizer**: RAdam + Lookahead
- **Sortino-based checkpointing**: Saves model when validation Sortino improves
- **Cosine Annealing**: Warm restarts scheduler

---

## Installation

```bash
# Clone repository
git clone https://github.com/alexrzk/ORION.git
cd ORION

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# For CUDA (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Quick Start

### 1. Fetch Data
```python
from orion.data.loader import load_and_prepare_data

df, aligner = load_and_prepare_data(years=5, use_cache=True)
print(f"Data shape: {df.shape}")
```

### 2. Train Model
```python
from orion.train_backtest import main

results = main()  # Full training pipeline
```

### 3. Or Run from CLI
```bash
python -m orion.train_backtest
```

---

## Project Structure

```
ORION/
├── orion/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py          # Data fetching & alignment
│   ├── math/
│   │   ├── __init__.py
│   │   └── fracdiff.py        # Fractional differencing
│   ├── models/
│   │   ├── __init__.py
│   │   └── hybrid.py          # TFT + QR-DQN architecture
│   └── train_backtest.py      # Training & backtesting pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Configuration

Key hyperparameters in `train_backtest.py`:

```python
config = {
    # Model
    'hidden_dim': 128,
    'num_heads': 4,
    'num_quantiles': 51,
    'lookback': 96,  # 8 hours at 5m
    
    # Training
    'num_epochs': 100,
    'batch_size': 256,
    'lr': 1e-4,
    
    # Trading
    'transaction_cost': 0.001,  # 0.1% per trade
    'risk_level': 'neutral',    # or 'averse', 'seeking'
}
```

---

## Mathematical Foundations

### Fractional Differencing
$(1 - B)^d X_t = \sum_{k=0}^{\infty} w_k X_{t-k}$

Where weights: $w_k = -w_{k-1} \cdot \frac{d - k + 1}{k}$

### Quantile Huber Loss
$\rho_\tau(\delta) = |\tau - \mathbb{1}(\delta < 0)| \cdot L_\kappa(\delta)$

### Sortino Ratio
$\text{Sortino} = \frac{R - R_f}{\sigma_d}$, where $\sigma_d = \sqrt{E[\min(R-R_f, 0)^2]}$

---

## Hardware Requirements

- **Minimum**: NVIDIA GPU with 8GB VRAM
- **Recommended**: RTX 4090 (24GB) for full batch sizes
- **Optimal**: Vast.ai with multi-GPU setup

---

## License

MIT License - Use at your own risk. Not financial advice.

---

## Acknowledgments

- Marcos López de Prado - *Advances in Financial Machine Learning*
- DeepMind - Distributional RL research
- Google - Temporal Fusion Transformer paper
