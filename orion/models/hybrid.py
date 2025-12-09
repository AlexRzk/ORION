# -*- coding: utf-8 -*-
"""
ORION Hybrid Model - Temporal Fusion Transformer + Quantile Regression DQN
===========================================================================

This module implements the core neural network architecture for ORION:
1. TFT Encoder: Extracts multi-scale temporal features with attention
2. QR-DQN Agent: Distributional RL for risk-aware action selection

ARCHITECTURE OVERVIEW:

    Multi-Timeframe Input (5m, 15m, 1h, 4h, 1d)
           ↓
    [Variable Selection Network (VSN)]
    - Learns which indicators matter per timestep
    - Soft attention over features within each timeframe
           ↓
    [Gated Residual Networks (GRN)]  
    - Filters noise via gating mechanism
    - Residual connections for gradient flow
           ↓
    [Multi-Head Attention (Interpretable)]
    - Attends across time (lookback window)
    - Multiple heads for different pattern types
           ↓
    [Context Embedding e_t]
    - Dense vector summarizing temporal state
           ↓
    [Quantile Regression DQN Head]
    - Outputs N quantiles per action
    - Captures full return distribution

MATHEMATICAL FOUNDATIONS:

1. GATED RESIDUAL NETWORK (GRN):
   GRN(x) = LayerNorm(x + GLU(W1 * ELU(W0 * x + b0) + b1))
   
   Where GLU (Gated Linear Unit):
   GLU(z) = σ(z_gate) ⊙ z_value
   
   The gate σ(z_gate) ∈ (0,1) controls information flow.
   This is inspired by LSTM gating but applied to feed-forward.

2. VARIABLE SELECTION NETWORK (VSN):
   Given feature matrix X ∈ R^{T×F}, VSN computes:
   
   v = Softmax(W * flatten(GRN(X)))  # Variable weights
   x̃ = Σ_i v_i * GRN(x_i)           # Weighted feature combination
   
   This learns which indicators are important at each step.

3. MULTI-HEAD ATTENTION:
   Attention(Q,K,V) = softmax(QK^T / √d_k) V
   MultiHead(X) = Concat(head_1, ..., head_h) W^O
   
   Where each head_i = Attention(XW_i^Q, XW_i^K, XW_i^V)

4. QUANTILE REGRESSION DQN:
   Instead of learning E[R], we learn the distribution of R.
   
   For action a, we output quantiles: {θ_τ(a) : τ ∈ {1/N, 2/N, ..., N/N}}
   
   Quantile Huber Loss:
   ρ_τ(u) = |τ - 𝟙(u<0)| * L_κ(u)
   
   Where L_κ(u) = {
       0.5 * u²           if |u| ≤ κ
       κ * (|u| - 0.5κ)   otherwise
   }

TENSOR SHAPES (annotated throughout):
- Batch: B = batch size
- Sequence: T = lookback length
- Features: F = total features per timestep
- Hidden: H = hidden dimension
- Heads: A = number of attention heads
- Actions: |A| = 4 (Long, Short, Hold, Close)
- Quantiles: N = 51 (default)

Author: ORION Quant Team
"""

import math
from typing import Tuple, Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ==============================================================================
# GATED RESIDUAL NETWORK (GRN)
# ==============================================================================
class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network - Core building block of TFT.
    
    ARCHITECTURE:
    x → Linear → ELU → Linear → GLU → + → LayerNorm → output
    └─────────────────────────────────────┘ (residual)
    
    MATH:
    η = ELU(W_1 * x + b_1)                    # First dense layer with ELU
    η̃ = W_2 * η + b_2                         # Second dense layer  
    z = GLU(η̃) = σ(η̃[:, :H]) ⊙ η̃[:, H:]     # Gated output
    GRN(x) = LayerNorm(x + Dropout(z))        # Residual + norm
    
    The GLU splits the output into two halves:
    - Gate half: passed through sigmoid to get values in (0,1)
    - Value half: the actual content
    The gate modulates the value, allowing the network to 
    "turn off" uninformative pathways.
    
    WHY ELU?
    ELU (Exponential Linear Unit) has benefits over ReLU:
    - Smooth around zero (better gradients)
    - Non-zero mean (reduces bias shift)
    - f(x) = x if x > 0, else α(e^x - 1)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            dropout: Dropout rate
            context_dim: Optional context vector dimension (for conditioning)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # First dense: input → hidden
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Optional context integration
        # If context is provided, it's added to the hidden representation
        # This allows conditioning the GRN on external information
        self.context_dim = context_dim
        if context_dim is not None:
            self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False)
        
        # Second dense: hidden → 2*output (for GLU)
        # We need 2x output because GLU splits into gate and value
        self.fc2 = nn.Linear(hidden_dim, output_dim * 2)
        
        # Residual projection (if dims don't match)
        self.residual_fc = None
        if input_dim != output_dim:
            self.residual_fc = nn.Linear(input_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor, shape (B, T, input_dim) or (B, input_dim)
            context: Optional context, shape (B, context_dim) or (B, T, context_dim)
            
        Returns:
            Output tensor, shape (B, T, output_dim) or (B, output_dim)
        """
        # Store original for residual
        residual = x
        
        # First dense + ELU
        # Shape: (B, [T,] hidden_dim)
        hidden = F.elu(self.fc1(x))
        
        # Add context if provided
        if context is not None and self.context_dim is not None:
            # Expand context to match hidden if needed
            if hidden.dim() == 3 and context.dim() == 2:
                context = context.unsqueeze(1).expand(-1, hidden.size(1), -1)
            hidden = hidden + self.context_fc(context)
        
        # Second dense (outputs 2x for GLU)
        # Shape: (B, [T,] output_dim * 2)
        glu_input = self.fc2(hidden)
        
        # GLU: split into gate and value
        # Shape of each: (B, [T,] output_dim)
        gate, value = glu_input.chunk(2, dim=-1)
        gated = torch.sigmoid(gate) * value
        
        # Dropout
        gated = self.dropout(gated)
        
        # Residual connection
        if self.residual_fc is not None:
            residual = self.residual_fc(residual)
        
        # Add & LayerNorm
        output = self.layer_norm(residual + gated)
        
        return output


# ==============================================================================
# VARIABLE SELECTION NETWORK (VSN)
# ==============================================================================
class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network - Learns importance of each input variable.
    
    ARCHITECTURE:
    For input X with F features:
    1. Transform each feature x_i through its own GRN
    2. Compute variable weights via softmax over a "flattened" GRN
    3. Output weighted combination of transformed features
    
    MATH:
    ξ_i = GRN_i(x_i)                          # Transform each feature
    v = Softmax(GRN_v([x_1, ..., x_F]))       # Compute weights
    x̃ = Σ_i v_i * ξ_i                         # Weighted combination
    
    WHY VSN?
    In multi-timeframe trading:
    - Not all indicators are useful all the time
    - RSI might matter in ranging markets (low ADX)
    - Trend indicators matter in trending markets (high ADX)
    VSN learns these dynamic importances from data.
    
    INTERPRETABILITY:
    The weights v_i are interpretable:
    - High v_i means indicator i is important for this timestep
    - Can analyze which features drive decisions
    """
    
    def __init__(
        self,
        num_features: int,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None
    ):
        """
        Args:
            num_features: Number of input features (F)
            input_dim: Dimension of each feature
            hidden_dim: GRN hidden dimension
            dropout: Dropout rate
            context_dim: Optional context for conditioning
        """
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        
        # Per-feature GRNs (transform each feature independently)
        # This allows learning feature-specific representations
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout,
                context_dim=context_dim
            )
            for _ in range(num_features)
        ])
        
        # Weight computation GRN
        # Input: concatenation of all features
        # Output: num_features weights (before softmax)
        self.weight_grn = GatedResidualNetwork(
            input_dim=num_features * input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_features,
            dropout=dropout,
            context_dim=context_dim
        )
        
    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features, shape (B, T, num_features, input_dim)
               OR (B, num_features, input_dim)
            context: Optional context vector
            
        Returns:
            Tuple of:
            - Selected output, shape (B, T, hidden_dim) or (B, hidden_dim)
            - Variable weights, shape (B, T, num_features) or (B, num_features)
        """
        # Handle both 3D and 4D input
        if x.dim() == 3:
            # (B, F, D) → (B, 1, F, D)
            x = x.unsqueeze(1)
            squeeze_time = True
        else:
            squeeze_time = False
        
        B, T, F, D = x.shape
        
        # Transform each feature through its GRN
        # List of (B, T, hidden_dim) tensors
        transformed = []
        for i in range(F):
            # x[:, :, i, :] has shape (B, T, D)
            feat_transformed = self.feature_grns[i](x[:, :, i, :], context)
            transformed.append(feat_transformed)
        
        # Stack transformed features: (B, T, F, hidden_dim)
        transformed_stack = torch.stack(transformed, dim=2)
        
        # Compute variable weights
        # Flatten input features: (B, T, F*D)
        flat_x = x.view(B, T, -1)
        
        # Get weights: (B, T, F)
        weights = self.weight_grn(flat_x, context)
        weights = F.softmax(weights, dim=-1)
        
        # Weighted sum: (B, T, hidden_dim)
        # weights: (B, T, F) → (B, T, F, 1)
        # transformed: (B, T, F, hidden_dim)
        output = (weights.unsqueeze(-1) * transformed_stack).sum(dim=2)
        
        if squeeze_time:
            output = output.squeeze(1)
            weights = weights.squeeze(1)
        
        return output, weights


# ==============================================================================
# INTERPRETABLE MULTI-HEAD ATTENTION
# ==============================================================================
class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention for temporal patterns.
    
    DIFFERENCE FROM STANDARD MHA:
    Standard: MultiHead = Concat(head_1, ..., head_h) W^O
    Ours: We share value projections across heads, making 
          attention patterns more interpretable.
    
    MATH:
    For each head h:
        Q_h = X W_h^Q,  K_h = X W_h^K,  V = X W^V (shared)
        head_h = softmax(Q_h K_h^T / √d_k) V
    
    Output = Mean(head_1, ..., head_h)  (average, not concat)
    
    WHY INTERPRETABLE?
    - Shared V means each head attends to the same representations
    - Averaging (not concat) means we can analyze each head's contribution
    - The attention weights show exactly what time steps are being used
    
    FOR TRADING:
    - Head 1 might attend to recent momentum (last few bars)
    - Head 2 might attend to support/resistance levels (longer lookback)
    - Head 3 might attend to volatility regimes
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: Dimension of input/output
            num_heads: Number of attention heads
            dropout: Dropout on attention weights
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Separate Q, K projections per head
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Shared V projection (key for interpretability)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with self-attention.
        
        Args:
            x: Input sequence, shape (B, T, hidden_dim)
            mask: Optional causal mask, shape (T, T)
                  mask[i,j] = 0 if i can attend to j, -inf otherwise
            
        Returns:
            Tuple of:
            - Output, shape (B, T, hidden_dim)
            - Attention weights, shape (B, num_heads, T, T)
        """
        B, T, _ = x.shape
        
        # Project Q, K, V
        # Shape: (B, T, hidden_dim)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head: (B, num_heads, T, head_dim)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores: (B, num_heads, T, T)
        # QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply causal mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Softmax → attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values: (B, num_heads, T, head_dim)
        attn_output = torch.matmul(attn_weights, V)
        
        # Average across heads (interpretable approach)
        # Shape: (B, T, head_dim)
        attn_output = attn_output.mean(dim=1)
        
        # Project back to hidden_dim
        # For interpretable version, we expand head_dim back to hidden_dim
        # Shape: (B, T, hidden_dim)
        if self.head_dim != self.hidden_dim:
            # Repeat to match hidden_dim
            attn_output = attn_output.repeat(1, 1, self.num_heads)
        
        output = self.out_proj(attn_output)
        
        return output, attn_weights


# ==============================================================================
# TEMPORAL FUSION TRANSFORMER (TFT) ENCODER
# ==============================================================================
class TFTEncoder(nn.Module):
    """
    Temporal Fusion Transformer Encoder.
    
    COMPLETE ARCHITECTURE:
    
    Input: Multi-timeframe features X ∈ R^{B × T × F}
           ↓
    [Feature Embedding] - Project raw features to hidden_dim
           ↓
    [Positional Encoding] - Add temporal position information
           ↓
    [Variable Selection Network] - Weight feature importance
           ↓
    [LSTM Encoder] - Capture local temporal patterns
           ↓
    [Gated Skip Connection] - Combine LSTM with static features
           ↓
    [Multi-Head Attention] - Attend across time
           ↓
    [Position-wise GRN] - Final transformation
           ↓
    Output: Context embedding e_t ∈ R^{hidden_dim}
    
    TENSOR FLOW (with shapes):
    
    raw_features: (B, T, num_features)
         ↓ embedding
    embedded: (B, T, hidden_dim)
         ↓ positional encoding
    positioned: (B, T, hidden_dim)
         ↓ VSN (reshape to per-feature)
    selected: (B, T, hidden_dim)
         ↓ LSTM
    lstm_out: (B, T, hidden_dim)
         ↓ gated skip
    enriched: (B, T, hidden_dim)
         ↓ attention
    attended: (B, T, hidden_dim)
         ↓ output GRN
    context: (B, T, hidden_dim)
    
    For RL, we typically use only the last timestep: e_t = context[:, -1, :]
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        lookback: int = 96  # 8 hours at 5m = 96 steps
    ):
        """
        Args:
            num_features: Number of input features per timestep
            hidden_dim: Hidden dimension throughout the network
            num_heads: Attention heads
            num_lstm_layers: LSTM layers for local patterns
            dropout: Dropout rate
            lookback: Maximum sequence length for attention
        """
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.lookback = lookback
        
        # Feature embedding: project each feature to hidden_dim
        self.feature_embedding = nn.Linear(num_features, hidden_dim)
        
        # Positional encoding (learnable)
        # Shape: (1, lookback, hidden_dim)
        self.positional_encoding = nn.Parameter(
            torch.randn(1, lookback, hidden_dim) * 0.02
        )
        
        # Variable Selection (treat each raw feature as a variable)
        # We reshape (B, T, F) → (B, T, F, 1) for VSN
        self.vsn = VariableSelectionNetwork(
            num_features=num_features,
            input_dim=1,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # LSTM for local temporal dependencies
        # LSTM captures short-range patterns (momentum, reversals)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            bidirectional=False  # Causal: only past information
        )
        
        # Gated skip connection
        # Allows network to bypass LSTM if direct path is better
        self.gated_skip = GatedResidualNetwork(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout
        )
        
        # Multi-head attention for long-range dependencies
        self.attention = InterpretableMultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Output GRN
        self.output_grn = GatedResidualNetwork(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout
        )
        
        # Layer norm for attention residual
        self.attn_layer_norm = nn.LayerNorm(hidden_dim)
        
        # Causal mask for attention
        # Lower triangular: position i can only attend to positions ≤ i
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((lookback, lookback), float('-inf')), diagonal=1)
        )
        
    def forward(
        self,
        x: Tensor,
        return_attention: bool = False
    ) -> Tuple[Tensor, Optional[Dict]]:
        """
        Forward pass.
        
        Args:
            x: Input features, shape (B, T, num_features)
            return_attention: If True, return attention weights for analysis
            
        Returns:
            Tuple of:
            - Context embedding, shape (B, hidden_dim) [last timestep]
            - Optional dict with attention weights and variable importance
        """
        B, T, F = x.shape
        
        # Clip to max lookback
        if T > self.lookback:
            x = x[:, -self.lookback:, :]
            T = self.lookback
        
        # 1. Variable Selection
        # Reshape for VSN: (B, T, F, 1) - each feature as separate input
        x_vsn = x.unsqueeze(-1)
        selected, var_weights = self.vsn(x_vsn)
        # selected: (B, T, hidden_dim)
        # var_weights: (B, T, num_features)
        
        # 2. Add positional encoding
        # Use learnable positional embeddings
        pos_enc = self.positional_encoding[:, :T, :]
        selected = selected + pos_enc
        
        # 3. LSTM for local patterns
        lstm_out, _ = self.lstm(selected)
        # lstm_out: (B, T, hidden_dim)
        
        # 4. Gated skip connection
        # Combines LSTM output with direct path
        enriched = self.gated_skip(lstm_out)
        
        # 5. Self-attention for long-range patterns
        # Causal mask ensures no future information leaks
        causal_mask = self.causal_mask[:T, :T]
        attn_out, attn_weights = self.attention(enriched, mask=causal_mask)
        
        # Residual + LayerNorm
        attended = self.attn_layer_norm(enriched + attn_out)
        
        # 6. Output GRN
        context = self.output_grn(attended)
        # context: (B, T, hidden_dim)
        
        # Extract last timestep for RL
        # This is the "current" state embedding
        final_context = context[:, -1, :]  # (B, hidden_dim)
        
        if return_attention:
            analysis = {
                'variable_weights': var_weights,  # (B, T, F)
                'attention_weights': attn_weights,  # (B, heads, T, T)
                'full_context': context  # (B, T, hidden_dim)
            }
            return final_context, analysis
        
        return final_context, None


# ==============================================================================
# QUANTILE REGRESSION DQN HEAD
# ==============================================================================
class QRDQNHead(nn.Module):
    """
    Quantile Regression DQN Head - Distributional RL.
    
    DISTRIBUTIONAL RL MOTIVATION:
    Standard DQN learns E[R|s,a] - the expected return.
    But in trading, we care about the DISTRIBUTION of returns:
    - Risk-averse: Avoid large losses (lower quantiles)
    - Risk-seeking: Maximize upside (upper quantiles)
    
    QR-DQN learns the full distribution via quantile regression.
    
    MATH:
    For N quantiles τ_1, ..., τ_N uniformly spaced in (0,1):
        τ_i = (i - 0.5) / N
    
    The network outputs θ_i(s,a) for each quantile and action.
    θ_i approximates the τ_i-th quantile of the return distribution.
    
    QUANTILE HUBER LOSS:
    For target z and prediction θ:
        δ = z - θ
        L_κ(δ) = 0.5*δ² if |δ| ≤ κ, else κ*(|δ| - 0.5κ)
        ρ_τ(δ) = |τ - 𝟙(δ<0)| * L_κ(δ)
    
    The asymmetric weight |τ - 𝟙(δ<0)| is key:
    - If τ=0.9 and δ<0 (underprediction): weight = |0.9 - 1| = 0.1 (low)
    - If τ=0.9 and δ>0 (overprediction): weight = |0.9 - 0| = 0.9 (high)
    This encourages the 0.9 quantile to be ABOVE 90% of outcomes.
    
    ACTION SELECTION:
    Two strategies:
    1. Risk-neutral: a* = argmax_a Mean(θ_1(a), ..., θ_N(a))
    2. Risk-averse: a* = argmax_a CVaR_α(θ(a)) = mean of lower α quantiles
    3. Risk-seeking: a* = argmax_a mean of upper quantiles
    """
    
    def __init__(
        self,
        input_dim: int,
        num_actions: int = 4,  # Long, Short, Hold, Close
        num_quantiles: int = 51,
        hidden_dim: int = 128
    ):
        """
        Args:
            input_dim: Dimension of input (TFT embedding)
            num_actions: Number of discrete actions
            num_quantiles: Number of quantiles to estimate (odd for median)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        
        # Quantile midpoints: τ_i = (i - 0.5) / N
        # These are the target quantile levels
        taus = torch.arange(1, num_quantiles + 1, dtype=torch.float32)
        taus = (taus - 0.5) / num_quantiles
        self.register_buffer('taus', taus)  # (N,)
        
        # Network: input → hidden → num_actions * num_quantiles
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_actions * num_quantiles)
        
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass - compute quantiles for all actions.
        
        Args:
            x: Input embedding, shape (B, input_dim)
            
        Returns:
            Quantiles for each action, shape (B, num_actions, num_quantiles)
        """
        # Two hidden layers with LayerNorm and ReLU
        h = F.relu(self.layer_norm1(self.fc1(x)))
        h = F.relu(self.layer_norm2(self.fc2(h)))
        
        # Output layer
        out = self.fc3(h)  # (B, num_actions * num_quantiles)
        
        # Reshape to (B, num_actions, num_quantiles)
        out = out.view(-1, self.num_actions, self.num_quantiles)
        
        return out
    
    def get_q_values(
        self,
        quantiles: Tensor,
        risk_level: str = 'neutral'
    ) -> Tensor:
        """
        Compute Q-values from quantiles based on risk preference.
        
        Args:
            quantiles: Shape (B, num_actions, num_quantiles)
            risk_level: One of:
                - 'neutral': Mean of all quantiles (standard Q-value)
                - 'averse': CVaR at 25% (mean of lower 25% quantiles)
                - 'seeking': Mean of upper 25% quantiles
                
        Returns:
            Q-values, shape (B, num_actions)
        """
        if risk_level == 'neutral':
            return quantiles.mean(dim=-1)
        
        N = self.num_quantiles
        if risk_level == 'averse':
            # CVaR: average of lower 25% quantiles
            # This represents expected value given we're in the worst 25% of cases
            cutoff = max(1, N // 4)
            return quantiles[:, :, :cutoff].mean(dim=-1)
        
        elif risk_level == 'seeking':
            # Mean of upper 25% quantiles
            cutoff = max(1, N // 4)
            return quantiles[:, :, -cutoff:].mean(dim=-1)
        
        else:
            raise ValueError(f"Unknown risk_level: {risk_level}")
    
    def compute_loss(
        self,
        current_quantiles: Tensor,
        target_quantiles: Tensor,
        actions: Tensor,
        kappa: float = 1.0
    ) -> Tensor:
        """
        Compute Quantile Huber Loss.
        
        MATH:
        For each quantile τ_i:
            δ_i = target - θ_i
            L_κ(δ_i) = 0.5*δ_i² if |δ_i| ≤ κ, else κ*(|δ_i| - 0.5κ)
            ρ_τ(δ_i) = |τ_i - 𝟙(δ_i<0)| * L_κ(δ_i)
        
        Total loss = (1/N) Σ_i ρ_{τ_i}(δ_i)
        
        Args:
            current_quantiles: Q(s,a) quantiles, shape (B, num_actions, N)
            target_quantiles: Target quantiles from Bellman, shape (B, N)
            actions: Actions taken, shape (B,)
            kappa: Huber loss threshold
            
        Returns:
            Scalar loss
        """
        B = current_quantiles.size(0)
        N = self.num_quantiles
        
        # Select quantiles for taken actions
        # current_quantiles: (B, A, N) → (B, N)
        action_idx = actions.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, N)
        current = current_quantiles.gather(1, action_idx).squeeze(1)  # (B, N)
        
        # Expand for pairwise comparison
        # current: (B, N, 1)  - each current quantile
        # target: (B, 1, N)   - each target quantile
        current_expanded = current.unsqueeze(-1)  # (B, N, 1)
        target_expanded = target_quantiles.unsqueeze(1)  # (B, 1, N)
        
        # Pairwise TD errors
        # δ[i,j] = target_j - current_i
        delta = target_expanded - current_expanded  # (B, N, N)
        
        # Huber loss
        abs_delta = delta.abs()
        huber = torch.where(
            abs_delta <= kappa,
            0.5 * delta ** 2,
            kappa * (abs_delta - 0.5 * kappa)
        )
        
        # Quantile weights
        # taus: (N,) → (1, N, 1) for broadcasting
        taus_expanded = self.taus.view(1, -1, 1)  # (1, N, 1)
        
        # Weight: |τ - 𝟙(δ<0)|
        # 𝟙(δ<0) is 1 if delta < 0, else 0
        indicator = (delta < 0).float()
        weights = (taus_expanded - indicator).abs()
        
        # Weighted loss
        loss = (weights * huber).sum(dim=-1).mean(dim=-1)  # (B,)
        
        return loss.mean()


# ==============================================================================
# COMPLETE TFT + QR-DQN MODEL
# ==============================================================================
class TFT_QR_DQN(nn.Module):
    """
    Complete Hybrid Model: TFT Encoder + QR-DQN Agent.
    
    FULL ARCHITECTURE:
    
    State s_t = [features_5m, features_15m, features_1h, features_4h, features_1d]
                           ↓
                    [TFT Encoder]
                           ↓
                  Context Embedding e_t
                           ↓
                    [QR-DQN Head]
                           ↓
            Quantiles for each action: Q(s_t, a) ~ {θ_τ}
                           ↓
                  Action Selection based on risk preference
    
    TRAINING:
    Uses standard DQN training with:
    - Experience replay buffer
    - Target network (updated periodically)
    - Quantile Huber loss
    
    The target for quantile i is:
        z_τ = r + γ * θ'_τ(s', a*)
    
    Where a* = argmax_a Mean(θ'(s', a)) from target network.
    
    ACTION SPACE:
    0 = Long:  Enter/hold long position
    1 = Short: Enter/hold short position  
    2 = Hold:  Maintain current position (no change)
    3 = Close: Close any open position
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_lstm_layers: int = 2,
        num_quantiles: int = 51,
        num_actions: int = 4,
        lookback: int = 96,
        dropout: float = 0.1
    ):
        """
        Args:
            num_features: Features per timestep
            hidden_dim: Model dimension
            num_heads: Attention heads
            num_lstm_layers: LSTM layers
            num_quantiles: Quantiles for distribution
            num_actions: Discrete actions
            lookback: Max sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        
        # TFT Encoder
        self.encoder = TFTEncoder(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout,
            lookback=lookback
        )
        
        # QR-DQN Head
        self.qr_head = QRDQNHead(
            input_dim=hidden_dim,
            num_actions=num_actions,
            num_quantiles=num_quantiles,
            hidden_dim=hidden_dim
        )
        
    def forward(
        self,
        x: Tensor,
        return_analysis: bool = False
    ) -> Tuple[Tensor, Optional[Dict]]:
        """
        Forward pass.
        
        Args:
            x: State features, shape (B, T, num_features)
            return_analysis: If True, return interpretability info
            
        Returns:
            Tuple of:
            - Quantiles, shape (B, num_actions, num_quantiles)
            - Optional analysis dict
        """
        # Encode state via TFT
        context, encoder_analysis = self.encoder(x, return_attention=return_analysis)
        
        # Get action quantiles
        quantiles = self.qr_head(context)
        
        if return_analysis:
            analysis = encoder_analysis or {}
            analysis['context'] = context
            return quantiles, analysis
        
        return quantiles, None
    
    def select_action(
        self,
        x: Tensor,
        risk_level: str = 'neutral',
        epsilon: float = 0.0
    ) -> Tuple[Tensor, Tensor]:
        """
        Select action using epsilon-greedy with risk preference.
        
        Args:
            x: State, shape (B, T, num_features) or (T, num_features)
            risk_level: Risk preference ('neutral', 'averse', 'seeking')
            epsilon: Exploration probability
            
        Returns:
            Tuple of:
            - Actions, shape (B,)
            - Q-values for selected actions, shape (B,)
        """
        # Handle single state
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        B = x.size(0)
        
        with torch.no_grad():
            quantiles, _ = self.forward(x)
            q_values = self.qr_head.get_q_values(quantiles, risk_level)
        
        # Epsilon-greedy
        if epsilon > 0:
            random_mask = torch.rand(B, device=x.device) < epsilon
            random_actions = torch.randint(
                0, self.num_actions, (B,), device=x.device
            )
            greedy_actions = q_values.argmax(dim=-1)
            actions = torch.where(random_mask, random_actions, greedy_actions)
        else:
            actions = q_values.argmax(dim=-1)
        
        # Get Q-values for selected actions
        selected_q = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        return actions, selected_q
    
    def compute_loss(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
        gamma: float = 0.99,
        target_net: Optional['TFT_QR_DQN'] = None
    ) -> Tensor:
        """
        Compute QR-DQN loss for a batch.
        
        BELLMAN UPDATE:
        For each quantile τ:
            target_τ = r + γ * (1 - done) * θ'_τ(s', a*)
        
        Where:
            a* = argmax_a Mean(θ'(s', a))  [greedy action from target]
        
        Args:
            states: Current states, (B, T, F)
            actions: Actions taken, (B,)
            rewards: Rewards received, (B,)
            next_states: Next states, (B, T, F)
            dones: Episode done flags, (B,)
            gamma: Discount factor
            target_net: Target network (if None, use self)
            
        Returns:
            Scalar loss
        """
        B = states.size(0)
        
        # Current Q-quantiles
        current_quantiles, _ = self.forward(states)  # (B, A, N)
        
        # Target computation
        with torch.no_grad():
            target = target_net if target_net is not None else self
            
            # Get target quantiles
            target_quantiles, _ = target.forward(next_states)  # (B, A, N)
            
            # Select best action (double DQN style)
            # Use current network to select action
            current_next_q, _ = self.forward(next_states)
            next_q_values = self.qr_head.get_q_values(current_next_q, 'neutral')
            best_actions = next_q_values.argmax(dim=-1)  # (B,)
            
            # Get quantiles for best action from target network
            best_action_idx = best_actions.unsqueeze(-1).unsqueeze(-1)
            best_action_idx = best_action_idx.expand(-1, 1, self.num_quantiles)
            next_quantiles = target_quantiles.gather(1, best_action_idx).squeeze(1)  # (B, N)
            
            # Bellman target for each quantile
            # target = r + γ * (1-done) * θ'
            target_vals = rewards.unsqueeze(-1) + \
                          gamma * (1 - dones.unsqueeze(-1)) * next_quantiles
        
        # Compute quantile loss
        loss = self.qr_head.compute_loss(
            current_quantiles,
            target_vals,
            actions
        )
        
        return loss


# ==============================================================================
# MODEL FACTORY
# ==============================================================================
def create_model(
    num_features: int,
    config: Optional[Dict] = None,
    device: str = 'cuda'
) -> TFT_QR_DQN:
    """
    Factory function to create TFT_QR_DQN model.
    
    Args:
        num_features: Number of input features
        config: Optional config dict with model hyperparameters
        device: Device to place model on
        
    Returns:
        Initialized model
    """
    default_config = {
        'hidden_dim': 128,
        'num_heads': 4,
        'num_lstm_layers': 2,
        'num_quantiles': 51,
        'num_actions': 4,
        'lookback': 96,
        'dropout': 0.1
    }
    
    if config:
        default_config.update(config)
    
    model = TFT_QR_DQN(
        num_features=num_features,
        **default_config
    )
    
    return model.to(device)


if __name__ == "__main__":
    # Test the model
    print("=" * 60)
    print("ORION TFT + QR-DQN Model Test")
    print("=" * 60)
    
    # Simulated input
    B = 32  # Batch size
    T = 96  # Lookback (8 hours at 5m)
    F = 50  # Features
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create model
    model = create_model(num_features=F, device=device)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Random input
    x = torch.randn(B, T, F, device=device)
    
    # Forward pass
    quantiles, analysis = model(x, return_analysis=True)
    print(f"\nInput shape: {x.shape}")
    print(f"Quantiles shape: {quantiles.shape}")  # (B, A, N)
    print(f"Context shape: {analysis['context'].shape}")
    print(f"Variable weights shape: {analysis['variable_weights'].shape}")
    
    # Action selection
    actions, q_values = model.select_action(x, risk_level='neutral')
    print(f"\nSelected actions shape: {actions.shape}")
    print(f"Q-values shape: {q_values.shape}")
    
    # Test loss computation
    rewards = torch.randn(B, device=device)
    dones = torch.zeros(B, device=device)
    next_states = torch.randn(B, T, F, device=device)
    
    loss = model.compute_loss(x, actions, rewards, next_states, dones)
    print(f"\nLoss: {loss.item():.4f}")
    
    print("\n✓ All tests passed!")
