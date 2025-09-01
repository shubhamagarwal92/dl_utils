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
  
# ----------- Top-p (Nucleus) Sampling -----------
# See the searchsorted one that is more optimized on PyTorch
def top_p_sampling(logits, p=0.9, temperature=1.0, use_searchsorted=True):
  # Sort logits
  sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
  sorted_logits = sorted_logits / temperature
  probs = F.softmax(sorted_logits, dim=-1)
  if use_searchsorted:
    # Efficient cutoff detection
    cumulative_probs = torch.cumsum(probs, dim=-1)
    # cutoff = torch.searchsorted(cumulative_probs, p, right=True)
    # cumulative_probs: [batch, vocab]
    # Makes a tensor of shape [batch, 1]
    # Fills every element with the scalar value p.
    p_tensor = torch.full((cumulative_probs.size(0), 1), p, device=logits.device)
    cutoff = torch.searchsorted(cumulative_probs, p_tensor, right=True).squeeze(-1) # [batch_size]
  
    batch_size, vocab_size = logits.shape
    arange = torch.arange(vocab_size, device=logits.device).unsqueeze(0).expand(batch_size, -1)  # (batch_size, vocab_size)
    # print(arange)
    cutoff_expanded = cutoff.unsqueeze(1).expand_as(arange) # [batch_size, vocab_size]
    # print(cutoff_expanded)
    mask = arange > cutoff_expanded
    """
    Eg: batch_size = 2; vocab_size = 6
    arange = [[0, 1, 2, 3, 4, 5],
      [0, 1, 2, 3, 4, 5]]  # shape [2, 6]
    cutoff_expanded = [[3, 3, 3, 3, 3, 3],
               [4, 4, 4, 4, 4, 4]]  # shape [2, 6]
    mask = [[False, False, False, False, True, True],
    [False, False, False, False, False, True]]
    """
  else:
    # Simpler mask-shift version
    cumulative_probs = torch.cumsum(probs, dim=-1)
    mask = cumulative_probs > p
    mask[..., 1:] = mask[..., :-1].clone()  # shift right
    mask[..., 0] = False
  
  # Mask logits & re-normalize
  filtered_logits = sorted_logits.masked_fill(mask, float('-inf'))
  filtered_probs = F.softmax(filtered_logits, dim=-1)
  
  # Sample
  sampled_idx = torch.multinomial(filtered_probs, num_samples=1)
  return torch.gather(sorted_indices, -1, sampled_idx)
  
  
  
  
