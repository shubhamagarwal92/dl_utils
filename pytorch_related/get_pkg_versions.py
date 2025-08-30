from importlib.metadata import version
import torch

pkgs = [
    "huggingface_hub",  # to download pretrained weights
    "tokenizers",       # to implement the tokenizer
    "torch",            # to implement the model
]
for p in pkgs:
    print(f"{p} version: {version(p)}")
  
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(device)
