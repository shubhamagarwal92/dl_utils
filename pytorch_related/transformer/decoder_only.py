# Through multiple iteration with GPT ;) 
# TODO: Have the mask as part of MHA class itself. 
# Use register buffer to avoid gradients
import torch
import torch.nn as nn
import math

# ==============================================================================
# Step 1: Scaled Dot-Product Attention
# This is the core attention mechanism, unchanged from the previous example.
# We will use it with a mask to ensure the decoder can only "see" previous tokens.
# ==============================================================================
def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Computes scaled dot-product attention.

    Args:
        query (torch.Tensor): Query tensor with shape (batch_size, num_heads, seq_len, d_k).
        key (torch.Tensor): Key tensor with shape (batch_size, num_heads, seq_len, d_k).
        value (torch.Tensor): Value tensor with shape (batch_size, num_heads, seq_len, d_v).
        mask (torch.Tensor, optional): Mask tensor. Defaults to None.

    Returns:
        torch.Tensor: Output tensor with shape (batch_size, num_heads, seq_len, d_v).
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attention_weights = nn.functional.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)

    return output, attention_weights

# ==============================================================================
# Step 2: Multi-Head Attention
# This is also largely the same as before, but in a decoder-only model,
# it is always a masked self-attention layer.
# ==============================================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1. Linear projections
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        # 2. Reshape for multi-head attention
        query = query.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3. Compute masked scaled dot-product attention
        attention_output, _ = scaled_dot_product_attention(query, key, value, mask)

        # 4. Concatenate heads and pass through final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.output_linear(attention_output)

        return output

# ==============================================================================
# Step 3: Position-wise Feed-Forward Network
# This is a standard component of each Transformer block.
# ==============================================================================
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

# ==============================================================================
# Step 4: Positional Encoding
# This adds information about the position of each token in the sequence.
# ==============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ==============================================================================
# Step 5: Decoder-Only Layer
# A single block in a decoder-only model. It consists of a masked multi-head
# attention sub-layer followed by a feed-forward network.
# ==============================================================================
class DecoderOnlyLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(DecoderOnlyLayer, self).__init__()
        self.masked_multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # 1. Masked multi-head attention on the input, with residual and layer norm
        attention_output = self.masked_multi_head_attention(x, x, x, mask)
        x = self.norm1(x + attention_output)

        # 2. Feed-forward network, with residual and layer norm
        feed_forward_output = self.feed_forward(x)
        x = self.norm2(x + feed_forward_output)

        return x

# ==============================================================================
# Step 6: The Full Decoder-Only Model
# This combines the embedding, positional encoding, and a stack of decoder layers.
# ==============================================================================
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len):
        super(DecoderOnlyTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_len)

        self.decoder_layers = nn.ModuleList([
            DecoderOnlyLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])

        self.output_linear = nn.Linear(d_model, vocab_size)

    def forward(self, trg, mask):
        # 1. Embed and apply positional encoding
        trg = self.positional_encoding(self.embedding(trg))

        # 2. Pass through all decoder layers
        decoder_output = trg
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, mask)

        # 3. Pass decoder output through the final linear layer
        output = self.output_linear(decoder_output)
        return output

# ==============================================================================
# Simple demonstration of how to use the Decoder-Only model
# ==============================================================================
if __name__ == '__main__':
    # Hyperparameters
    vocab_size = 1000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_len = 100
    batch_size = 32

    # Dummy data
    trg_seq_len = 60
    trg_tokens = torch.randint(0, vocab_size, (batch_size, trg_seq_len))

    # Create a look-ahead mask for the decoder.
    # It is a square matrix where the upper triangle is set to 0.
    trg_mask = torch.tril(torch.ones(trg_seq_len, trg_seq_len)).bool()
    trg_mask = trg_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, trg_seq_len, trg_seq_len)

    # Instantiate and run the model
    model = DecoderOnlyTransformer(vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len)

    print("Decoder-Only Transformer model initiated.")
    print("Input target tensor shape:", trg_tokens.shape)

    output = model(trg_tokens, trg_mask)

    print("Output tensor shape:", output.shape)
    print("The model output is a tensor of logits for the target vocabulary.")
