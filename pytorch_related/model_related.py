import torch
import torch.nn as nn

# A linear layer that takes an input of size 20 and produces an output of size 10
linear_layer = nn.Linear(20, 10)

# Sample sequential module
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
)


