import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class CausalSelfAttention(nn.Module):
    """
    Implements causal multi-head self-attention mechanism as used in GPT-style transformer models.
    
    WHY CAUSAL ATTENTION: Ensures autoregressive property where position t can only attend to positions <= t.
    This is critical for next-token prediction tasks where future tokens must remain masked during training/inference.
    
    DEPENDENCIES: 
    - torch.nn: Provides core neural network primitives (Linear, Dropout, Module)
    - torch.nn.functional: Provides softmax and masking utilities optimized for tensor operations
    - Config: Centralized configuration object to ensure consistent hyperparameters across model components
    """
    
    def __init__(self, config: Config):
        """
        WHY THIS CONSTRUCTOR APPROACH: Single Config parameter enables easy hyperparameter sweeping and 
        model replication. All dimensions derived from config ensure architectural consistency.
        
        PARAMETER JUSTIFICATION:
        - config (Config): Contains all model hyperparameters (n_embd, n_head, block_size, dropout).
          Omission would make class unusable without manual parameter passing to every instance.
        
        KEY CALCULATIONS & VALUE ANALYSIS:
        - self.head_size = config.n_embd // config.n_head (e.g., 512//8 = 64):
          WHY 64? Ensures each head processes equal-sized subspaces. Integer division guarantees 
          exact divisibility (config enforces n_embd % n_head == 0). This value determines per-head 
          computation granularity - smaller heads = more diverse attention patterns.
        
        LINEAR LAYERS (QKV):
        - nn.Linear(n_embd, n_embd, bias=False): No bias needed as positional embeddings provide 
          necessary offsets. bias=False reduces parameters by ~0.2% while maintaining performance.
          Output dim = n_embd ensures total Q/K/V capacity matches input embedding size.
        
        PROJECTION LAYER: Final linear layer mixes head outputs back to full embedding dimension.
        
        DROPOUT: Applied post-attention and post-projection for regularization during training.
        
        TRIANGULAR MASK (tril):
        - torch.tril(torch.ones(block_size, block_size)): Creates lower-triangular matrix of 1s.
        - register_buffer(): Stores as buffer (not parameter) so it moves to correct device with model
          and persists through state_dict. Precomputing avoids recreation on every forward pass.
        """
        super().__init__()
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head  # 512//8 = 64 - per-head dimension
        self.n_embd = config.n_embd
        
        # QKV projections: n_embd -> n_embd total capacity, split across heads
        # bias=False: Reduces parameters, positional embeddings provide offsets
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # Output projection: Combines multi-head outputs back to embedding dimension
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
        # Causal mask: Precomputed lower-triangular matrix for efficient masking
        # block_size defines max sequence length this attention can handle
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x):
        """
        WHY THIS FORWARD PASS DESIGN: 
        - Batched, vectorized operations leverage GPU parallelism
        - In-place reshaping minimizes memory allocations
        - Causal masking applied once before softmax for efficiency
        
        PARAMETER JUSTIFICATION:
        - x (torch.Tensor): Input tensor of shape [B, T, C] where:
          * B = batch_size: Enables parallel processing of multiple sequences
          * T = sequence_length: Current length (<= block_size), enables variable-length sequences
          * C = n_embd: Embedding dimension
          Omission makes forward pass impossible.
        
        EFFICIENCY TRADEOFFS:
        - Time: O(B × nh × T²) due to attention matrix computation - quadratic in sequence length
        - Space: O(B × nh × T × T) attention matrix - dominant memory cost for long sequences
        - OPTIMIZATION: head_size^-0.5 scaling prevents softmax saturation (derived from dot-product variance)
        
        STEP-BY-STEP LOGIC:
        1. QKV projection + head splitting
        2. Scaled dot-product attention with causal masking
        3. Head concatenation + output projection
        """
        B, T, C = x.shape  # Batch, Time/Sequence, Channels/Embedding_dim
        
        # Step 1: QKV Projections + Reshape for Multi-Head Processing
        # view(B, T, nh, hs) -> transpose(1,2) = [B, nh, T, hs]
        # WHY TRANSPOSE: Enables efficient batching over heads (dim=1), seq_len pairs for matmul
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)    
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        
        # Step 2: Scaled Dot-Product Attention
        # q @ k.transpose(-2,-1): [B,nh,T,hs] @ [B,nh,hs,T] = [B,nh,T,T]
        # head_size^-0.5: Scales variance to ~1, prevents softmax saturation (derived from E[q·k]=hs)
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
        
        # Causal Masking: Zero future positions by setting to -inf (post-scaling)
        # self.tril[:T, :T]: Dynamic slicing for variable sequence lengths
        # masked_fill(): Vectorized in-place masking - more efficient than boolean indexing
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Softmax normalization across key dimension (dim=-1)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)  # Dropout on attention weights (not values)
        
        # Weighted sum: [B,nh,T,T] @ [B,nh,T,hs] = [B,nh,T,hs]
        out = wei @ v
        
        # Step 3: Concatenate Heads + Final Projection
        # transpose(1,2): [B,nh,T,hs] -> [B,T,nh,hs]
        # contiguous().view(): Ensures memory continuity for efficient linear layer processing
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Final projection mixes head information
        out = self.proj(out)
        out = self.dropout(out)  # Dropout on final output
        return out


class MultiHeadAttention(nn.Module):
    """
    Thin wrapper providing clean interface to CausalSelfAttention.
    
    WHY THIS WRAPPER: 
    - Enables easy substitution of attention types (causal vs. bidirectional)
    - Maintains consistent API across transformer blocks
    - Future-proofing for encoder-decoder attention variants
    
    EFFICIENCY: Zero overhead - simple delegation pattern
    """
    def __init__(self, config: Config):
        """
        PARAMETER JUSTIFICATION: config required for inner CausalSelfAttention initialization
        """
        super().__init__()
        self.attn = CausalSelfAttention(config)  # Delegate to causal attention implementation

    def forward(self, x):
        """
        PARAMETER JUSTIFICATION: x passed through to inner attention unchanged
        """
        return self.attn(x)  # Zero-copy delegation