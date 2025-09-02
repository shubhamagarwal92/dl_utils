# Through multiple iterations with GPT to get the final code ;) 
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

# ----------- Readable and better Top-p (Nucleus) Sampling -----------
def top_p_sampling_v2(logits, p: float):
    """
    logits: [batch_size, vocab_size]
    p: nucleus probability threshold (0 < p <= 1)
    """
    batch_size, vocab_size = logits.shape
    # Step 1: sort logits (descending)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    # Step 2: compute cumulative probs
    probs = F.softmax(sorted_logits, dim=-1) # [b, vocab]
    cumulative_probs = torch.cumsum(probs, dim=-1)  # [b, vocab]
    # Step 3: find cutoff index for each batch element
    p_tensor = torch.full((batch_size, 1), p, device=logits.device)
    cutoff = torch.searchsorted(cumulative_probs, p_tensor, right=True).squeeze(-1)  # [batch_size] bcoz of squeeze
    # Step 4: create mask (broadcasting, no expand_as)    
    mask = torch.arange(vocab_size, device=logits.device).unsqueeze(0) > cutoff.unsqueeze(1) # [batch, vocab] # broadcasting
    # same as:  
    # mask = torch.arange(vocab_size, device=logits.device)[None, :] > cutoff[:, None]  # [batch, vocab]
    # Step 5: mask logits
    filtered_logits = sorted_logits.masked_fill(mask, float('-inf'))
    # Step 6: renormalize & sample
    filtered_probs = F.softmax(filtered_logits, dim=-1)
    sampled_idx = torch.multinomial(filtered_probs, num_samples=1)  # [batch, 1]
    # Step 7: map back to original vocab indices
    return torch.gather(sorted_indices, -1, sampled_idx)



# ----------- Autoregressive Generation Loop -----------
def generate(model, input_ids, max_new_tokens=20, strategy="greedy", temperature=1.0, k=50, p=0.9, eos_token_id=None):
    batch_size = input_ids.size(0)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

    # AR is always per token. Cant avoid it unless using vLLM.
    for _ in range(max_new_tokens):
        logits = model(input_ids)  # [batch, seq_len, vocab_size]
        logits = logits[:, -1, :]  # last token logits

        if strategy == "greedy":
            next_token = greedy_sampling(logits)
        elif strategy == "temperature":
            next_token = temperature_sampling(logits, temperature)
        elif strategy == "top-k":
            next_token = top_k_sampling(logits, k, temperature)
        elif strategy == "top-p":
            next_token = top_p_sampling(logits, p, temperature)
        else:
            raise ValueError("Unknown strategy")

        # If EOS handling
        if eos_token_id is not None:
            next_token = next_token.squeeze(-1)
            next_token = torch.where(finished, torch.full_like(next_token, eos_token_id), next_token)
            next_token = next_token.unsqueeze(-1)

            # Update finished status
            finished |= (next_token.squeeze(-1) == eos_token_id)

            # If all sequences finished, break early
            if finished.all():
                input_ids = torch.cat([input_ids, next_token], dim=1)
                break

        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids


# Example usage (dummy model)
class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
    def forward(self, x):
        batch, seq_len = x.shape
        return torch.randn(batch, seq_len, self.vocab_size)


if __name__ == "__main__":
    vocab_size = 10
    eos_token_id = 9
    model = DummyModel(vocab_size)
    input_ids = torch.zeros((2, 1), dtype=torch.long)

    print("Greedy:", generate(model, input_ids, strategy="greedy", eos_token_id=eos_token_id))
    print("Temperature:", generate(model, input_ids, strategy="temperature", temperature=0.7, eos_token_id=eos_token_id))
    print("Top-k:", generate(model, input_ids, strategy="top-k", k=5, eos_token_id=eos_token_id))
    print("Top-p:", generate(model, input_ids, strategy="top-p", p=0.9, eos_token_id=eos_token_id))
  
  
