import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from pathlib import Path
import time
import glob
from tqdm import tqdm
from config import get_config
from dataset import load_datasets, BatchSampler
from model import MiniGPT

def find_txt_file():
    """Find .txt file in current directory"""

    txt_files = list(Path('.').glob('*.txt'))

    if txt_files:
        # Prefer input.txt
        for f in txt_files:
            if f.name == 'input.txt':
                print(f"📄 Found: {f} ({f.stat().st_size/1e6:.2f} MB)")
                return str(f)

        # Otherwise take first .txt
        f = txt_files[0]
        print(f"📄 Using: {f} ({f.stat().st_size/1e6:.2f} MB)")
        return str(f)

    print("❌ No .txt files found!")
    print("💡 Put a .txt file in this folder (same as train.py)")
    return None
def main():
    config = get_config()
    
    # Auto-find data file
    config.data_path = "input.txt"
    if not config.data_path:
        return
    
    if config.device == 'cpu':
        config.compile = False
        print("⚠️  CPU - torch.compile disabled")
    
    print(f"🚀 Training on {config.device}")
    
    # Load data
    train_dataset, val_dataset = load_datasets(config)
    train_sampler = BatchSampler(train_dataset, config.batch_size, config.device)
    val_sampler = BatchSampler(val_dataset, config.batch_size, config.device)
    
    print(f"✅ Train: {len(train_dataset):,} | Val: {len(val_dataset):,} tokens")
    
    # Model
    model = MiniGPT(config).to(config.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 {total_params/1e6:.1f}M params")
    
    if config.compile and config.device == 'cuda':
        print("⚡ Compiling...")
        model = torch.compile(model)
    
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = GradScaler('cuda' if config.device == 'cuda' else 'cpu')
    
    model.train()
    best_val_loss = float('inf')
    
    @torch.no_grad()
    def estimate_loss():
        model.eval()
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = val_sampler.get_batch(eval_mode=True)
            with autocast(device_type='cuda' if config.device == 'cuda' else 'cpu'):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        model.train()
        return losses.mean().item()
    
    pbar = tqdm(range(config.max_iters), desc="Training")
    tick = time.time()
    
    for iter_num in pbar:
        xb, yb = train_sampler.get_batch()
        
        with autocast(device_type='cuda' if config.device == 'cuda' else 'cpu'):
            logits, loss = model(xb, yb)
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        if iter_num % config.eval_interval == 0:
            val_loss = estimate_loss()
            dt = time.time() - tick
            ppl = torch.exp(torch.tensor(val_loss))
            
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'val': f'{val_loss:.3f}',
                'ppl': f'{ppl:.0f}'
            })
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config.__dict__,
                    'best_val_loss': val_loss
                }, config.model_path)
                print(f"\n💾 Best saved: {val_loss:.3f}")
            
            tick = time.time()
    
    print(f"\n🎉 Complete! Model: {config.model_path}")

if __name__ == "__main__":
    main()