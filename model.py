import torch
import torch.nn as nn
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

# =================================self-Attention==========================================

class Head(nn.Module):
    def __init__(self, embedding, head_size, block_size):

        super().__init__()

        self.key = nn.Linear(embedding, head_size, bias = False)
        self.query = nn.Linear(embedding, head_size, bias = False)
        self.value = nn.Linear(embedding, head_size, bias = False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.head = head_size

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):

        B, T, C = x.shape

        # Generating k, q, v
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Calculating attention scores
        scores = q @ k.transpose(-2, -1)

        # Scale attention scores
        scores = scores * (self.head ** -0.5)

        # Apply casual masking
        scores = scores.masked_fill(self.tril[:T,:T] == 0, float('-inf'))

        # Apply Softmax
        weights = F.softmax(scores, dim=-1)

        # Apply dropout to attention weights
        weights = self.dropout(weights)

        # Weighted aggregation
        out = weights @ v

        return out

# ==============================Multi-Head-Attention=======================================

class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, embedding, head_size, block_size):

        super().__init__()

        self.head = nn.ModuleList([
            Head(embedding, head_size, block_size)
            for _ in range(n_heads)
        ])

        self.projection = nn.Linear(n_heads * head_size, embedding)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Run all attention heads independently
        outputs = [head(x) for head in self.head]

        # Concatenate their outputs
        out = torch.cat(outputs, dim = -1)

        # Apply output projection
        out = self.projection(out)

        out = self.dropout(out)

        return out

# ==============================Feed-Forward-Network=======================================

class FeedForwardNetwork(nn.Module):

    def __init__(self, embedding, hidden_dim):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding),
            nn.Dropout(0.1)
        )

    def forward(self, x):

        return self.network(x)

# ======================================Block==============================================

class Block(nn.Module):

    def __init__(self, embedding, n_heads, block_size):

        super().__init__()

        head_size = embedding // n_heads

        # Multihead Attention
        self.sa = MultiHeadAttention(
            n_heads,
            embedding,
            head_size,
            block_size
        )

        # Feed Forward Network
        self.ffwd = FeedForwardNetwork(embedding, 4 * embedding)

        # LayerNorm
        self.ln1 = nn.LayerNorm(embedding)
        self.ln2 = nn.LayerNorm(embedding)

    def forward(self, x):

        # Attention + Residual
        x = x + self.sa(self.ln1(x))

        # Feed forward + Residual
        x = x + self.ffwd(self.ln2(x))

        return x

# ===============================GPT Language Model========================================

class GPTLanguageModel(nn.Module):

    def __init__(self,vocab_size, embedding, block_size, n_layers, n_heads):

        super().__init__()

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embedding)

        # Position Embedding
        self.position_embedding = nn.Embedding(block_size, embedding)

        self.blocks = nn.Sequential(*[Block(embedding, n_heads, block_size) for _ in range(n_layers)])

        # Final LayerNorm
        self.ln_f = nn.LayerNorm(embedding)

        # Language Model Head
        self.lm_head = nn.Linear(embedding, vocab_size)

    def forward(self, idx):
        B, T = idx.shape

        # Token Embedding
        tok_embedding = self.token_embedding(idx)

        # Position Embedding
        pos = torch.arange(T, device = device)
        pos_embedding = self.position_embedding(pos)

        # Combine Embedding
        x = tok_embedding + pos_embedding

        # Transformer Blocks
        x = self.blocks(x)

        # Final Normalization
        x = self.ln_f(x)

        # Generate Logits
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):

        for _ in range(max_new_tokens):

            # Keep only the most recent context
            idx_cond = idx[:, -self.position_embedding.num_embeddings:]

            # Get predictions
            logits = self(idx_cond)

            # Get predictions for the final position
            logits = logits[:, -1, :]

            # Apply temperature
            logits = logits / temperature

            # Convert logits into probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample the next character
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append the generated character
            idx = torch.cat((idx, idx_next), dim=1)

        return idx