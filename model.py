import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from transformer_block import TransformerBlock

class MiniGPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        
        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layer)]
        )
        
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Share weights
        self.lm_head.weight = self.token_embedding.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @torch.no_grad()
    def estimate_loss(self, train_data, val_data, eval_iters: int):
        self.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            sampler = train_data if split == 'train' else val_data
            for k in range(eval_iters):
                X, Y = sampler.get_batch(eval_mode=True)
                with torch.amp.autocast('cuda' if self.config.device == 'cuda' else 'cpu'):
                    _, loss = self(X, Y)
                losses[k] = loss.item()
            out = losses.mean()
            print(f"{split} loss: {out:.4f} | ppl: {torch.exp(out):.2f}")
        self.train()

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Embeddings
        tok_emb = self.token_embedding(idx)           # (B,T,C)
        pos_emb = self.position_embedding(torch.arange(T, device=self.config.device))  # (T,C)
        x = tok_emb + pos_emb                         # (B,T,C)
        
        # Transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T,vocab)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                 targets.view(-1), ignore_index=-1)
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx