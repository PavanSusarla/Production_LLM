import torch.nn as nn
from config import Config
from attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN) as used in transformer architectures.
    
    WHY FEEDFORWARD LAYER: Provides non-linear transformation capacity and increases model expressivity.
    Each position processes independently, enabling parallel computation across sequence length.
    
    WHY THIS EXPANSION RATIO (4x): Empirical finding from "Attention is All You Need" paper.
    Creates a "bottleneck" expansion: input → 4×expansion → projection back. This ratio balances
    capacity (more parameters) with regularization (information bottleneck effect).
    
    DEPENDENCIES: 
    - nn.Sequential: Chains layers with automatic forward propagation
    - nn.GELU: Gaussian Error Linear Unit - smoother than ReLU, better gradient flow
    - nn.Dropout: Regularization applied post-projection (not between layers)
    """
    
    def __init__(self, config: Config):
        """
        PARAMETER JUSTIFICATION:
        - config (Config): Provides n_embd (embedding dim) and dropout rate.
          Omission breaks dimension consistency across model.
        
        ARCHITECTURE DETAILS & VALUE ANALYSIS:
        - Expansion: n_embd → 4×n_embd → n_embd
          WHY 4×? Provides ~4× parameter increase per block while maintaining output dim.
          E.g., 512 → 2048 → 512 = ~2.1M params per FFN
        - GELU: Approximates probabilistic neuron firing, empirically superior to ReLU
        - Dropout: Applied only on final output for computational efficiency
        """
        super().__init__()
        self.net = nn.Sequential(
            # Expansion layer: Creates high-dimensional representation
            nn.Linear(config.n_embd, 4 * config.n_embd),
            
            # Non-linearity: GELU preferred over ReLU for smoother gradients
            nn.GELU(), 
            
            # Projection back to embedding dimension (information bottleneck)
            nn.Linear(4 * config.n_embd, config.n_embd),
            
            # Dropout on output only - more efficient than intermediate dropout
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        """
        PARAMETER JUSTIFICATION:
        - x (torch.Tensor): [B, T, n_embd] - applies identical transformation to each position
        
        EFFICIENCY: O(B × T × n_embd × 4×n_embd) = O(B × T × n_embd²) linear in sequence length,
        enabling efficient parallel computation across positions.
        """
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Single transformer decoder block implementing the pre-norm residual architecture.
    
    WHY PRE-NORM RESIDUALS: 
    1. LayerNorm before sub-layers (pre-norm) stabilizes training vs post-norm
    2. Residual connections prevent vanishing gradients in deep networks
    3. Dual normalization (attn + ffn) handles different transformation scales
    
    WHY THIS ORDER (Attn → FFN): Attention captures dependencies, FFN provides position-wise capacity.
    
    DEPENDENCIES: MultiHeadAttention imported for self-attention mechanism.
    """
    
    def __init__(self, config: Config):
        """
        PARAMETER JUSTIFICATION:
        - config (Config): Ensures all components share identical dimensions
        
        LAYER NORMALIZATION DETAILS:
        - nn.LayerNorm(n_embd): Normalizes across embedding dimension only (not sequence)
          WHY n_embd only? Preserves sequence dependencies while stabilizing embedding magnitudes
        - Two separate LayerNorms: One per sub-block (attn/ffn) prevents interference
        
        SUBMODULES:
        - MultiHeadAttention: Causal self-attention for autoregressive modeling
        - FeedForward: Position-wise MLP for capacity
        """
        super().__init__()
        # Pre-norm LayerNorms: Stabilize before attention/FFN
        self.ln1 = nn.LayerNorm(config.n_embd)  # Before attention
        self.ln2 = nn.LayerNorm(config.n_embd)  # Before FFN
        
        # Attention sub-block
        self.attn = MultiHeadAttention(config)
        
        # Feed-forward sub-block
        self.ffwd = FeedForward(config)

    def forward(self, x):
        """
        PARAMETER JUSTIFICATION:
        - x (torch.Tensor): [B, T, n_embd] input sequence
        
        PRE-NORM RESIDUAL ARCHITECTURE:
        x → LN → Attention → x + Attention(x)
              ↓
        x → LN → FFN → x + FFN(x)
        
        WHY RESIDUALS (x + sublayer(x)):
        1. Enables gradient flow through identity mapping
        2. Allows blocks to learn incremental refinements
        3. Prevents vanishing gradients in deep stacks
        
        WHY PRE-NORM SPECIFICALLY:
        - LayerNorm(x) stabilizes input distribution before non-linearities
        - Residual skip provides clean gradient path around sub-layers
        - Empirically more stable than post-norm for deep transformers
        
        EFFICIENCY TRADEOFFS:
        - Time: Dominated by attention O(T²) + FFN O(n_embd²)
        - Space: Minimal overhead from LayerNorm (O(n_embd) per position)
        - Parallelizable across batch/sequence dimensions
        """
        # Attention residual block: Pre-norm + causal self-attention
        x = x + self.attn(self.ln1(x))  # Residual connection
        
        # FFN residual block: Pre-norm + position-wise feedforward
        x = x + self.ffwd(self.ln2(x))  # Residual connection
        
        return x