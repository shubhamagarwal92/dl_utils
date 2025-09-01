import torch
import torch.nn.functional as F


# ----------- Greedy Sampling -----------
def greedy_sampling(logits):
  # softmax is monotonic, so avoid unnecessary computation  
  return torch.argmax(logits, dim=-1, keepdim=True) # [bs, 1]

# ----------- Temperature Sampling -----------
def temperature_sampling(logits, temperature=1.0):
  if temperature <= 0:
    raise ValueError("Temperature must be positive")
  scaled_logits = logits / temperature  # broadcasting
  probs = F.softmax(scaled_logits, dim=-1)
  return torch.multinomial(probs, num_samples=1)
  # multinomial already preserves the "extra" dimension automatically.


  
  
