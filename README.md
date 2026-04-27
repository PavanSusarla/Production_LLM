# MiniGPT - Production-Ready GPT from Scratch

## Overview
This project implements a complete, production-grade GPT model (MiniGPT) from scratch using PyTorch. It includes a full training pipeline with BPE tokenization, causal self-attention, multi-head attention, transformer blocks, mixed precision training, gradient clipping, and model checkpointing. The system supports both training on custom text datasets and interactive generation.

## Technical Logic
The architecture follows the GPT-2 design pattern: token and position embeddings feed into a stack of transformer decoder blocks, each containing causal self-attention (with triangular masking) followed by a feed-forward network, both wrapped in residual connections and layer normalization. Training uses next-token prediction on random context windows from the dataset, with evaluation via perplexity. Generation employs top-k sampling with temperature scaling for coherent text completion.

## Why This Matters
Building LLMs from scratch reveals the engineering decisions behind billion-parameter models. This implementation demonstrates production techniques like weight sharing, mixed precision (AMP), torch.compile optimization, efficient batch sampling, and robust checkpointing—skills essential for scaling to larger models.

## How to Run
1. Place a text file named `input.txt` in the project directory
2. Run training: `python train.py`
3. Generate text: `python generate.py`
4. Customize hyperparameters via command line arguments

## Expected Outcomes
- ~45M parameter model trained in ~25 minutes on consumer GPU
- Validation perplexity drops from ~10k to ~3k+ on small datasets
- Coherent text generation after 5k iterations
- Saved checkpoint `minigpt.pt` ready for inference

## Key Takeaways
- Causal attention masking enables autoregressive generation
- Proper divisibility (embd_size ÷ n_heads) prevents dimension errors
- Mixed precision + gradient accumulation scales training efficiently
- Weight sharing between embedding and LM head saves ~2x memory

## Next Steps
- Scale to larger datasets (1GB+ text)
- Add learning rate scheduling and warmup
- Implement grouped-query attention for efficiency
- Build LoRA fine-tuning pipeline
