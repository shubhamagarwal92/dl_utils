# Taken from
# https://github.com/allenai/allennlp/blob/master/allennlp/modules/openai_transformer.py

import torch.nn as nn
import torch
import math
from typing import NamedTuple, List


class Swish(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
