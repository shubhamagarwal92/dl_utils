import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 1. Scaled Dot-Product Attention
# -----------------------------
class ScaledDotProductAttention(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.scale = head_dim ** 0.5

    def forward(self, Q, K, V, mask=None):
        # Q,K,V: (B, T, d)
        scores = Q @ K.transpose(-2, -1) / self.scale   # (B, T, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)             # (B, T, T)
        return weights @ V                              # (B, T, d)


# -----------------------------
# 2. Multi-Head Attention
# -----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # projections
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attention = ScaledDotProductAttention(self.head_dim)

    def forward(self, x, mask=None):
        B, T, D = x.size()
        H, d = self.num_heads, self.head_dim

        # project
        Q = self.W_q(x).view(B, T, H, d).transpose(1, 2)   # (B, H, T, d)
        K = self.W_k(x).view(B, T, H, d).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, d).transpose(1, 2)

        # apply attention per head
        out = self.attention(Q, K, V, mask)   # (B, H, T, d)

        # combine heads
        out = out.transpose(1, 2).contiguous().view(B, T, D)   # (B, T, D)
        return self.out_proj(out)                             # (B, T, D)


# -----------------------------
# 3. Feed-Forward Network
# -----------------------------
class FeedForward(nn.Module):
    def __init__(self, hidden_dim, ff_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# 4. Transformer Block
# -----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, ff_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention
        x = self.norm1(x + self.dropout(self.attn(x, mask)))
        # Feed-forward
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


# -----------------------------
# 5. Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-torch.log(torch.tensor(10000.0)) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# -----------------------------
# 6. Transformer (Decoder-only example)
# -----------------------------
class Transformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, ff_dim, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, max_len)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, mask=None):
        # Embedding + position
        x = self.embed(x)                # (B, T, D)
        x = self.pos_enc(x)              # (B, T, D)

        # Pass through N layers
        for layer in self.layers:
            x = layer(x, mask)

        return self.fc_out(x)            # (B, T, vocab_size)



# ---- Hyperparameters ----
vocab_size = 1000     # toy vocab
hidden_dim = 64
num_layers = 2
num_heads = 4
ff_dim = 256
seq_len = 10
batch_size = 2

# ---- Create model ----
model = Transformer(
    vocab_size=vocab_size,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    num_heads=num_heads,
    ff_dim=ff_dim,
    max_len=seq_len
)

# ---- Dummy input ----
x = torch.randint(0, vocab_size, (batch_size, seq_len))  # (B, T)

# ---- Causal mask for decoder ----
mask = torch.tril(torch.ones(seq_len, seq_len))  # (T, T)
mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # (B, T, T)

# ---- Forward pass ----
logits = model(x, mask=mask)   # (B, T, vocab_size)

print("Input shape: ", x.shape)
print("Logits shape:", logits.shape)  # expect (2, 10, 1000)


targets = torch.randint(0, vocab_size, (batch_size, seq_len))
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
