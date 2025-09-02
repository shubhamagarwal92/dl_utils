import torch
import numpy as np

def kl_divergence_npy(p, q, eps=1e-12):
    """
    Compute the KL divergence D(P || Q).
    Args:
        p (np.ndarray): Probability distribution P.
        q (np.ndarray): Probability distribution Q.
        eps (float): Small constant to avoid log(0).
    Returns:
        float: KL divergence.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    # normalize distributions
    p = p / np.sum(p)
    q = q / np.sum(q)
    # avoid division by zero
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.sum(p * np.log(p / q))


def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL divergence D(P || Q) for batched inputs.
    Args:
        p (torch.Tensor): Probability distribution P (..., num_classes).
        q (torch.Tensor): Probability distribution Q (..., num_classes).
        eps (float): Small constant to avoid log(0).
    Returns:
        torch.Tensor: KL divergence for each batch.
    """
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    return torch.sum(p * torch.log(p / q), dim=-1)

# Example
p = torch.tensor([[0.4, 0.6],[0.7, 0.3]])
q = torch.tensor([[0.5, 0.5],[0.6, 0.4]])
# Or
# p = torch.softmax(torch.randn(2, 2), dim=-1)
# q = torch.softmax(torch.randn(2, 2), dim=-1)

print("KL(P || Q):", kl_divergence(p, q))


def info_nce_loss(z1, z2, temperature=0.07):
    """
    z1, z2: (batch, dim) embeddings
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.size(0), device=z1.device)

    return F.cross_entropy(logits, labels)

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    anchor, positive, negative: (batch, dim)
    """
    d_pos = F.pairwise_distance(anchor, positive)
    d_neg = F.pairwise_distance(anchor, negative)
    return torch.mean(torch.clamp(d_pos - d_neg + margin, min=0.0))

def sigmoid_contrastive_loss(z1, z2):
    """
    z1, z2: (batch, dim)
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    logits = z1 @ z2.T   # (batch, batch)
    labels = torch.eye(z1.size(0), device=z1.device)  # positives on diagonal

    loss = F.binary_cross_entropy_with_logits(logits, labels)
    return loss

def ppo_loss(old_log_probs, new_log_probs, advantages, eps=0.2):
    """
    old_log_probs, new_log_probs, advantages: (batch,)
    """
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
    return -torch.mean(torch.min(ratio * advantages, clipped))

def cross_entropy_loss(logits, targets):
    """
    logits: (batch, num_classes)
    targets: (batch,) indices
    """
    return F.cross_entropy(logits, targets)

def seq_cross_entropy_loss(logits, targets, mask=None):
    """
    logits: (batch, seq_len, vocab_size)
    targets: (batch, seq_len) indices
    mask: (batch, seq_len) 1 for valid tokens, 0 for padding
    """
    log_probs = F.log_softmax(logits, dim=-1)
    loss_per_token = F.nll_loss(log_probs.transpose(1, 2), targets, reduction='none')
    # Here the mask is defined in the opposite way
    if mask is not None:
        loss_per_token = loss_per_token * mask
    return loss_per_token.sum(dim=1) / (mask.sum(dim=1) + 1e-12) if mask is not None else loss_per_token.mean(dim=1)
