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
  return torch.multinomial(probs, num_samples=1)  # [bs, 1]
  # multinomial already preserves the "extra" dimension automatically.

# ----------- Top-k Sampling -----------
def top_k_sampling(logits, k=50, temperature=1.0):
  values, indices = torch.topk(logits, k, dim=-1)
  probs = F.softmax(values / temperature, dim=-1)  # Convert in prob distribution
  sampled_idx = torch.multinomial(probs, num_samples=1)  # [bs, 1]
  return torch.gather(indices, -1, sampled_idx) 
  # Also need to "gather" / map it to original indices (over Vocab). 
  # Top k changed the order earlier but preserved the mapping
  

  
  
