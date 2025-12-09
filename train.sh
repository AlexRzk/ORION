#!/bin/bash
# ==============================================================================
# O.R.I.O.N. Training Script for Vast.ai
# ==============================================================================
# This script simplifies training on Vast.ai GPU instances.
# 
# Usage:
#   chmod +x train.sh
#   ./train.sh              # Full training with defaults
#   ./train.sh --quick      # Quick test run (5 epochs)
#   ./train.sh --resume     # Resume from checkpoint
#
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  O.R.I.O.N. - Omniscient Risk & Investment Optimization   ║"
echo "║                   Training Pipeline                        ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="${SCRIPT_DIR}/venv"
CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoints"
DATA_CACHE="${SCRIPT_DIR}/data_cache"

# Default training mode
MODE="full"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            MODE="quick"
            shift
            ;;
        --resume)
            MODE="resume"
            shift
            ;;
        --help)
            echo "Usage: ./train.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick   Quick test run (5 epochs, small batch)"
            echo "  --resume  Resume training from last checkpoint"
            echo "  --help    Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================
echo -e "${YELLOW}[1/5] Checking environment...${NC}"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python version: $python_version"

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n1)
    echo -e "  GPU: ${GREEN}$gpu_name ($gpu_memory)${NC}"
else
    echo -e "  GPU: ${RED}Not detected${NC}"
    echo -e "${YELLOW}Warning: No GPU found. Training will be slow.${NC}"
fi

# ==============================================================================
# VIRTUAL ENVIRONMENT
# ==============================================================================
echo -e "${YELLOW}[2/5] Setting up virtual environment...${NC}"

if [ ! -d "$VENV_DIR" ]; then
    echo "  Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
echo "  Activated: $VIRTUAL_ENV"

# ==============================================================================
# INSTALL DEPENDENCIES
# ==============================================================================
echo -e "${YELLOW}[3/5] Installing dependencies...${NC}"

pip install --quiet --upgrade pip

# Install PyTorch with CUDA support
if command -v nvidia-smi &> /dev/null; then
    # Check CUDA version
    cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "  CUDA version: $cuda_version"
    
    # Install appropriate PyTorch
    pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    pip install --quiet torch torchvision torchaudio
fi

# Install other requirements
pip install --quiet -r "${SCRIPT_DIR}/requirements.txt"

echo -e "  ${GREEN}Dependencies installed${NC}"

# ==============================================================================
# CREATE DIRECTORIES
# ==============================================================================
echo -e "${YELLOW}[4/5] Creating directories...${NC}"

mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$DATA_CACHE"

echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Data cache: $DATA_CACHE"

# ==============================================================================
# START TRAINING
# ==============================================================================
echo -e "${YELLOW}[5/5] Starting training...${NC}"
echo ""

cd "$SCRIPT_DIR"

case $MODE in
    quick)
        echo -e "${BLUE}Mode: QUICK TEST (5 epochs)${NC}"
        python3 -c "
import sys
sys.path.insert(0, '.')
from orion.train_backtest import OrionTrainer, load_and_prepare_data, preprocess_data, train_test_split_ts, validate_data
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ORION.QuickTest')

# Quick test config
config = {
    'years_of_history': 1,  # Less data for quick test
    'cache_dir': './data_cache',
    'hidden_dim': 64,  # Smaller model
    'num_heads': 2,
    'num_lstm_layers': 1,
    'num_quantiles': 21,
    'lookback': 48,
    'dropout': 0.1,
    'num_epochs': 5,  # Quick test
    'batch_size': 128,
    'steps_per_epoch': 100,
    'updates_per_step': 2,
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'gamma': 0.99,
    'max_grad_norm': 1.0,
    'use_ranger': True,
    'scheduler_t0': 2,
    'scheduler_t_mult': 2,
    'scheduler_eta_min': 1e-6,
    'initial_epsilon': 1.0,
    'final_epsilon': 0.1,
    'epsilon_decay': 0.9,
    'buffer_size': 10_000,
    'target_update_freq': 50,
    'transaction_cost': 0.001,
    'risk_level': 'neutral',
    'backtest_freq': 2,
    'checkpoint_dir': './checkpoints'
}

logger.info('Loading data (1 year for quick test)...')
df, aligner = load_and_prepare_data(years=1, use_cache=True)
df = preprocess_data(df, aligner)
validate_data(df, 'Quick Test Data')
train_df, val_df, test_df = train_test_split_ts(df)

trainer = OrionTrainer(config)
trainer.setup(train_df, val_df, test_df)

logger.info('Starting quick training...')
history = trainer.train()

logger.info('Running final backtest...')
trainer.backtest()

logger.info('Quick test complete!')
"
        ;;
    resume)
        echo -e "${BLUE}Mode: RESUME FROM CHECKPOINT${NC}"
        # TODO: Implement resume logic
        echo -e "${RED}Resume mode not yet implemented. Running full training.${NC}"
        python3 -m orion.train_backtest
        ;;
    full)
        echo -e "${BLUE}Mode: FULL TRAINING${NC}"
        python3 -m orion.train_backtest
        ;;
esac

# ==============================================================================
# SUMMARY
# ==============================================================================
echo ""
echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                   Training Complete!                       ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo "Results saved to:"
echo "  - Best validation model: $CHECKPOINT_DIR/best_model.pt"
echo "  - Best profitable model: $CHECKPOINT_DIR/best_profitable_model.pt"
echo "  - Final results: $CHECKPOINT_DIR/final_results.json"
echo ""
echo "To view the best model metrics:"
echo "  cat $CHECKPOINT_DIR/best_metrics.json"
echo "  cat $CHECKPOINT_DIR/best_profitable_metrics.json"
