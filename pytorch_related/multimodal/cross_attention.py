# Taken from https://medium.com/@jakubstrawadev/why-cross-attention-is-the-secret-sauce-of-multimodal-models-f8ec77fc089b

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):

    def __init__(self, dim_query, dim_kv, dim_out):
        super().__init__()
        self.query_proj = nn.Linear(dim_query, dim_out)
        self.key_proj = nn.Linear(dim_kv, dim_out)
        self.value_proj = nn.Linear(dim_kv, dim_out)
        self.out_proj = nn.Linear(dim_out, dim_out)

    def forward(self, queries, keys, values, mask=None):
        Q = self.query_proj(queries)   # (batch, query_len, dim_out)
        K = self.key_proj(keys)        # (batch, key_len, dim_out)
        V = self.value_proj(values)    # (batch, key_len, dim_out)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1)**0.5)  # (batch, query_len, key_len)
        attn_weights = F.softmax(scores, dim=-1)                      # (batch, query_len, key_len)
        output = torch.bmm(attn_weights, V)                           # (batch, query_len, dim_out)
        return self.out_proj(output), attn_weights

#these are random values
batch_size = 2
seq_len_text = 5
seq_len_image = 10
text_embedding_dim = 32
image_embedding_dim = 64
cross_attn_dim = 128  # Common attention space dimension

# Instantiate the model
cross_attention = CrossAttention(
    dim_query=text_embedding_dim,
    dim_kv=image_embedding_dim,
    dim_out=cross_attn_dim
)

# Generate random embeddings (simulating real embeddings)
text_embeddings = torch.randn(batch_size, seq_len_text, text_embedding_dim)
image_embeddings = torch.randn(batch_size, seq_len_image, image_embedding_dim)

# Cross-attention: Text queries attend to Image keys/values
output, attn_weights = cross_attention(
    queries=text_embeddings,
    keys=image_embeddings,
    values=image_embeddings
)
