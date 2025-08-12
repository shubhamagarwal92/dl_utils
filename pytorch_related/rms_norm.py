import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x):
        # Root mean square along last dimension
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x / rms)

# Example
x = torch.randn(2, 3, 4)
rmsnorm = RMSNorm(4)
output = rmsnorm(x)
print(output.shape)  # torch.Size([2, 3, 4])
