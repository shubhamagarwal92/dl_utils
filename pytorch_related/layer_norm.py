import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(MyLayerNorm, self).__init__()
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        # Mean & variance over the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)

        # Normalize
        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        return self.gamma * x_hat + self.beta

# Example usage
x = torch.randn(2, 3, 4)  # [batch, seq, hidden]
ln = LayerNorm(4)
output = ln(x)
print(output.shape)  # torch.Size([2, 3, 4])
