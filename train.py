import torch
import torch.nn.functional as F
import math
import os

from config import batch_size, block_size, embedding, n_heads, n_layers, hidden_dim, learning_rate, min_lr, epochs, warmup_steps, eval_iters, vocab_size
import model
from model import GPTLanguageModel

from tokenizer import (build_tokenizer, encode, decode, save_tokenizer)

device = "cuda" if torch.cuda.is_available() else "cpu"

with open('input/input.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()
print("File Opened Successfully")

chars = sorted(list(set(''.join(text))))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

save_tokenizer(stoi)
print("Saved Tokenizer File")
#Encoder
def encode(s):
    return [stoi[c] for c in s]

#Decoder
def decode(l): # Inputs list of integers outputs string
    return ''.join(itos[i] for i in l)

# ===================================Dataset Preparation===================================
# Train-validation split
data = torch.tensor(encode(text), dtype = torch.long)
split_index = int(0.9*len(data))

train_data = data[:split_index]
val_data = data[split_index:]

# Creating Batches
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size - 1, (batch_size, ))
    x = torch.stack([data[i : i+block_size] for i in ix])
    y = torch.stack([data[i+1 : i+block_size+1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x,y # Dimension : (batch_size, block_size) 

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, targets = get_batch(split)
            logits = model(x)
            loss = F.cross_entropy(logits.transpose(1, 2),targets)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def lr_lambda(step):

    if step < warmup_steps:
        return (step + 1) / warmup_steps

    progress = (step - warmup_steps) / (epochs - warmup_steps)

    return min_lr / learning_rate + (
        1 - min_lr / learning_rate
    ) * 0.5 * (1 + math.cos(math.pi * progress))


# ====================================Training-Loop========================================

model = GPTLanguageModel(vocab_size, embedding, block_size, n_layers, n_heads).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = 0.01)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda)

print("Number of Parameters =",sum(p.numel() for p in model.parameters()))
print("Training Started ...")
for i in range(epochs):

    if i % 500 == 0:
            losses = estimate_loss()
            print(f"For {i:4d}th iteration, Training Loss : {losses['train']:.4f}, Validation Loss : {losses['val']:.4f}")

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': i,
                'config': {
                    'vocab_size': vocab_size,
                    'embedding': embedding,
                    'block_size': block_size,
                    'n_layers': n_layers,
                    'n_heads': n_heads
                }
            }, f'saved_parameters/checkpoint{i}.pth')

    # Getting training batch
    x, targets = get_batch('train')

    # Forward Propagation
    logits = model(x)

    # Calculate Loss
    loss = F.cross_entropy(logits.transpose(1, 2), targets)

    # Backpropagation
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Graditent clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Update parameters
    optimizer.step()

    # Update Learning Rate
    scheduler.step()

print("Training Completed")

torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': {
                'vocab_size': vocab_size,
                'embedding': embedding,
                'block_size': block_size,
                'n_layers': n_layers,
                'n_heads': n_heads
            }
            }, 'saved_parameters/checkpoint_final.pth')