import torch
import torch.nn.functional as F

def resize_pos_embed(pos_embed, old_hw, new_hw):
    """
    pos_embed: [1, N_old, D]
    old_hw: (H_old, W_old)
    new_hw: (H_new, W_new)
    """
    B, N, D = pos_embed.shape
    H_old, W_old = old_hw
    H_new, W_new = new_hw
    
    pos = pos_embed.reshape(1, H_old, W_old, D).permute(0, 3, 1, 2)  # [1, D, H_old, W_old]
    pos_resized = F.interpolate(pos, size=(H_new, W_new), mode="bicubic", align_corners=False)
    pos_resized = pos_resized.permute(0, 2, 3, 1).reshape(1, H_new*W_new, D)
    return pos_resized


pos_embed = torch.randn(1, 14*14, 768)   # learned ViT pos embed
new_pos_embed = resize_pos_embed(pos_embed, (14,14), (16,16))
print(new_pos_embed.shape)  # [1, 256, 768]
