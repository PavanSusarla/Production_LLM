import torch
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """
    Centralized configuration system for MiniGPT (Production-ready).

    WHY DATACLASS:
    ---------------------------------------------------------
    - Cleaner than manual __init__
    - Automatic attribute assignment
    - Easy debugging (auto-generated __repr__)
    - Type hints improve reliability

    DESIGN PHILOSOPHY:
    ---------------------------------------------------------
    - Provide strong defaults (works out-of-the-box)
    - Allow CLI overrides (flexibility)
    - Validate constraints early (fail fast)

    TRADE-OFF:
    ---------------------------------------------------------
    + Clean and maintainable
    - Slight overhead vs simple class
    """


    # =========================
    # MODEL ARCHITECTURE
    # =========================

    # Vocabulary size
    # ---------------------------------------------------------
    # WHY 50257:
    # - Matches GPT-2 tokenizer
    # - Ensures compatibility with BPE tokenizer
    vocab_size: int = 50257


    # Context window (sequence length)
    # ---------------------------------------------------------
    # WHY 256:
    # - Balanced memory vs performance
    #
    # NOTE (IMPORTANT):
    # - Attention complexity = O(T²)
    block_size: int = 256


    # Embedding dimension
    # ---------------------------------------------------------
    # WHY 512:
    # - Larger than beginner models → better capacity
    #
    # CONSTRAINT:
    # Must be divisible by n_head
    n_embd: int = 512


    # Number of attention heads
    # ---------------------------------------------------------
    # WHY 8:
    # - Each head = 64 dims (512 / 8)
    #
    # TRADE-OFF:
    # More heads → more parallel attention patterns
    n_head: int = 8


    # Number of transformer layers
    # ---------------------------------------------------------
    # WHY 6:
    # - Mid-sized model (not too shallow, not too heavy)
    n_layer: int = 6


    # Dropout rate
    # ---------------------------------------------------------
    # WHY 0.1:
    # - Standard Transformer regularization
    dropout: float = 0.1


    # Bias in linear layers
    # ---------------------------------------------------------
    # WHY False:
    # - Modern GPTs often remove bias (simplifies model)
    # - Slight performance + efficiency gain
    bias: bool = False


    # =========================
    # TRAINING
    # =========================

    batch_size: int = 32


    # Learning rate
    # ---------------------------------------------------------
    # WHY 6e-4:
    # - Slightly higher than standard 3e-4
    # - Works well for smaller models
    learning_rate: float = 6e-4


    # Total iterations
    max_iters: int = 5000


    # Evaluation frequency
    eval_interval: int = 200


    # Number of eval batches
    # ---------------------------------------------------------
    # WHY:
    # - Stabilizes evaluation metric
    eval_iters: int = 200


    # Weight decay (L2 regularization)
    # ---------------------------------------------------------
    # WHY 0.1:
    # - Helps prevent overfitting
    weight_decay: float = 0.1


    # Gradient clipping
    # ---------------------------------------------------------
    # WHY 1.0:
    # - Prevents exploding gradients
    grad_clip: float = 1.0


    # Gradient accumulation
    # ---------------------------------------------------------
    # WHY:
    # - Simulates larger batch size
    #
    # EXAMPLE:
    # batch_size=32, accumulation=4 → effective batch=128
    gradient_accumulation_steps: int = 1


    # =========================
    # DATA
    # =========================

    data_path: str = "input.txt"


    # Train split ratio
    # ---------------------------------------------------------
    # WHY 0.9:
    # - Standard 90/10 split
    train_split: float = 0.9


    # Max data size
    # ---------------------------------------------------------
    # WHY 1M:
    # - Prevents loading huge datasets accidentally
    #
    # Optional:
    # - Allows None (use full dataset)
    max_data_size: Optional[int] = 1_000_000


    # =========================
    # GENERATION
    # =========================

    max_new_tokens: int = 256


    # Temperature
    # ---------------------------------------------------------
    # WHY 0.8:
    # - Slightly creative but still coherent
    temperature: float = 0.8


    # Top-k sampling
    # ---------------------------------------------------------
    # WHY 50:
    # - Common balance between diversity and quality
    top_k: int = 50


    # =========================
    # SYSTEM
    # =========================

    # Device (set dynamically)
    device: str = field(init=False)


    # Automatic Mixed Precision
    use_amp: bool = field(init=False)


    # Torch compile (PyTorch 2.x optimization)
    # ---------------------------------------------------------
    # WHY True:
    # - Speeds up model execution
    # - Reduces Python overhead
    #
    # TRADE-OFF:
    # - Initial compile time overhead
    compile: bool = True


    def __post_init__(self):
        """
        Post-initialization hook.

        WHY THIS EXISTS:
        ---------------------------------------------------------
        - Some parameters depend on runtime environment
        - Need validation after object creation
        """

        # Step 1: Device detection
        # ---------------------------------------------------------
        # WHY:
        # - Automatically use GPU if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


        # Step 2: AMP toggle
        # ---------------------------------------------------------
        # WHY:
        # - AMP only works effectively on GPU
        self.use_amp = self.device == 'cuda'


        # Step 3: Validate divisibility constraints
        # ---------------------------------------------------------
        # IMPORTANT CONSTRAINTS:

        # ❌ INCORRECT CHECK (your version)
        # block_size % n_head → NOT required

        # ✅ CORRECT:
        # n_embd must be divisible by n_head
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )


        # WHY THIS MATTERS:
        # - Multi-head attention splits embedding:
        #   head_size = n_embd / n_head
        #
        # If not divisible:
        # - Tensor shape mismatch → runtime error



def get_config() -> Config:
    """
    Parses CLI arguments and returns validated config.

    WHY THIS FUNCTION EXISTS:
    ---------------------------------------------------------
    - Allows runtime configuration via command line
    - Keeps defaults but enables overrides

    EXAMPLE:
    ---------------------------------------------------------
    python train.py --batch_size 64 --n_layer 8
    """

    parser = argparse.ArgumentParser(
        description="MiniGPT - Production LLM"
    )


    # CLI Arguments
    # ---------------------------------------------------------
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--block_size', type=int, default=256)
    parser.add_argument('--n_embd', type=int, default=512)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--max_iters', type=int, default=5000)

    # NOTE:
    # CLI uses 'lr' but config uses 'learning_rate'
    # This mismatch needs mapping
    parser.add_argument('--lr', type=float, default=6e-4)

    parser.add_argument('--data_path', type=str, default='input.txt')
    parser.add_argument('--model_path', type=str, default='minigpt.pt')

    # Boolean flag pattern
    # ---------------------------------------------------------
    parser.add_argument('--compile', action='store_true', default=True)
    parser.add_argument('--no-compile', dest='compile', action='store_false')


    args = parser.parse_args()


    # Create default config
    config = Config()


    # Override config with CLI args
    # ---------------------------------------------------------
    for key, value in vars(args).items():

        if value is not None:

            # ⚠️ IMPORTANT FIX:
            # Map CLI 'lr' → 'learning_rate'
            if key == 'lr':
                setattr(config, 'learning_rate', value)
            else:
                setattr(config, key, value)


    # Debug print
    # ---------------------------------------------------------
    print(
        "✅ Config validated:",
        config.n_embd, "embd,",
        config.n_head, "heads,",
        config.block_size, "ctx"
    )


    return config