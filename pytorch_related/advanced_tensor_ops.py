import torch

########### Indexing ################
# Example 1 (1D tensor)
x = torch.tensor([10, 20, 30, 40])

# 1). torch.index_select(input, dim, index)
# fancy indexing operator
idx = torch.tensor([0, 2, 4])
y = torch.index_select(x, 0, idx)
print(y)   # tensor([10, 30, 50])

# 2). Indexing
y = x[idx]   # advanced indexing
print(y)     # tensor([10, 30, 50])

# 3). torch.gather(input, dim, index)
# Unlike index_select, where index must be 1D, 
# in gather the index tensor can be multi-dimensional 
# which makes it much more powerful.

# Example 2 (2D tensor)
x = torch.arange(12).view(3, 4)  # shape (3,4)
idx = torch.tensor([0, 2])
# Advanced indexing (select rows 0 and 2)
y1 = x[idx, :]
# index_select (same thing)
y2 = torch.index_select(x, 0, idx)
print(torch.equal(y1, y2))  # True


# 2D row selection
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
idx = torch.tensor([[0, 2],   # pick col0, col2 from row0
                    [1, 0],   # pick col1, col0 from row1
                    [2, 1]])  # pick col2, col1 from row2
y = torch.gather(x, 1, idx)

print(y)
# tensor([[1, 3],
#         [5, 4],
#         [9, 8]])


# Top-k in Attention
logits = torch.tensor([[0.1, 0.7, 0.2],
                       [0.5, 0.4, 0.6]])
topk_vals, topk_idx = torch.topk(logits, k=2, dim=-1)
selected = torch.gather(logits, -1, topk_idx)


# Rule of thumb:
# Use index_select when you just want to slice along one axis with a list of indices.
# Use gather when you need per-row / per-batch custom indexing (e.g. in transformer attention, beam search, top-k sampling).

# Index select is useful in RoPE: 
even_idx = torch.arange(0, dim, 2)
odd_idx  = torch.arange(1, dim, 2)

x1 = x_rot.index_select(-1, even_idx)  # take even dims
x2 = x_rot.index_select(-1, odd_idx)   # take odd dims


###########  ################
