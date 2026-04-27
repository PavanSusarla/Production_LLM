import tiktoken
import numpy as np
from typing import List, Tuple
from pathlib import Path


class BPETokenizer:
    """
    Byte Pair Encoding (BPE) Tokenizer using tiktoken (GPT-2 compatible).

    WHY THIS CLASS EXISTS:
    ---------------------------------------------------------
    Converts raw text ↔ token IDs for model consumption.

    Compared to character-level tokenization:
    - Produces fewer tokens (efficient sequences)
    - Captures subword structure (better semantics)
    - Matches real-world LLM tokenization

    DEPENDENCY JUSTIFICATION:
    ---------------------------------------------------------
    tiktoken:
    - Highly optimized (Rust backend)
    - Exact GPT-2 encoding compatibility
    - Faster than Python-based tokenizers

    numpy:
    - (Currently unused in logic)
    - Likely intended for future stats/analysis
    - Can be safely removed for cleaner code

    pathlib:
    - Modern, OS-independent file handling
    - Preferred over raw string paths
    """


    def __init__(self, encoding_name: str = "gpt2"):
        """
        Initializes tokenizer with a specific encoding.

        PARAMETER:
        ---------------------------------------------------------
        encoding_name : str (default = "gpt2")

        WHY THIS PARAMETER:
        - Allows flexibility to switch tokenizers
        - Supports future extension (e.g., "cl100k_base")

        DEFAULT CHOICE ("gpt2"):
        - Vocabulary size: 50257
        - Widely used and well-tested

        WHAT HAPPENS INTERNALLY:
        ---------------------------------------------------------
        tiktoken.get_encoding():
        - Loads BPE merges + vocabulary rules
        - Precompiled for speed

        COST:
        - Small initialization overhead (acceptable)
        """

        self.enc = tiktoken.get_encoding(encoding_name)

        # Vocabulary size
        # ---------------------------------------------------------
        # WHY:
        # - Useful for model configuration alignment
        # - Ensures embedding layer matches tokenizer
        self.vocab_size = self.enc.n_vocab


    def encode(self, text: str) -> List[int]:
        """
        Converts raw text → list of token IDs.

        PARAMETER:
        ---------------------------------------------------------
        text : str
            Input string

        RETURNS:
        ---------------------------------------------------------
        List[int] : token IDs

        WHY THIS FUNCTION:
        - Model requires numerical input
        - Converts human-readable text into machine format

        TIME COMPLEXITY:
        - O(n), where n = length of text

        EDGE CASES:
        - Empty string → returns empty list
        - Rare unicode handled internally by tiktoken
        """

        return self.enc.encode(text)


    def decode(self, tokens: List[int]) -> str:
        """
        Converts token IDs → readable text.

        PARAMETER:
        ---------------------------------------------------------
        tokens : List[int]

        RETURNS:
        ---------------------------------------------------------
        str : decoded text

        WHY THIS FUNCTION:
        - Required for interpreting model outputs

        TIME COMPLEXITY:
        - O(n), where n = number of tokens

        EDGE CASES:
        - Invalid tokens → may raise error
        - Partial tokens → imperfect reconstruction
        """

        return self.enc.decode(tokens)


    @staticmethod
    def get_stats(data_path: str) -> Tuple[int, int, float]:
        """
        Computes dataset statistics.

        PARAMETERS:
        ---------------------------------------------------------
        data_path : str
            Path to text dataset

        RETURNS:
        ---------------------------------------------------------
        Tuple:
            (num_characters, num_tokens, compression_ratio)

        WHY THIS FUNCTION EXISTS:
        ---------------------------------------------------------
        Helps analyze tokenization efficiency.

        METRICS:
        ---------------------------------------------------------
        1. Character count:
           - Raw dataset size

        2. Token count:
           - Actual model input size

        3. Compression ratio:
           = characters / tokens

        INTERPRETATION:
        ---------------------------------------------------------
        Higher ratio → better compression
        Example:
            1000 chars → 250 tokens → ratio = 4.0

        WHY IMPORTANT:
        - Impacts training speed and memory
        - Helps choose tokenizer strategy

        PROCESS:
        ---------------------------------------------------------
        1. Read file
        2. Tokenize
        3. Compute stats

        TIME COMPLEXITY:
        - O(n) for reading + encoding

        EDGE CASE:
        - Empty file → division by zero risk
        """

        # Read dataset
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()


        # Initialize tokenizer
        tokenizer = BPETokenizer()


        # Encode text
        tokens = tokenizer.encode(text)


        # Compute stats
        num_chars = len(text)
        num_tokens = len(tokens)

        # Avoid division by zero
        compression_ratio = (
            num_chars / num_tokens
            if num_tokens > 0 else 0.0
        )


        return num_chars, num_tokens, compression_ratio