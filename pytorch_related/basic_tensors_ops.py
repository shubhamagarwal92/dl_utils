import torch

# Creates a tensor from data, like a Python list or NumPy array
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data, dtype=torch.float32, requires_grad=True)


# Tensor of a specified shape filled with zeros or ones. 
# Useful for initializing parameters or temporary buffers
zeros_tensor = torch.zeros(2, 3)
ones_tensor = torch.ones(5)


# tensor with values drawn from a standard normal distribution
random_weights = torch.randn(10, 5)


## View and reshape
# view returns a new tensor with the same data as the original but with a different shape
x = torch.randn(4, 4)
y = x.view(16)
z = x.reshape(2, 8)


## Size and shape
# Shape is preferred in modern PyTorch for its conciseness
x = torch.randn(2, 3, 4)
# Get the full size
print(x.size())  # Outputs: torch.Size([2, 3, 4])
# Get the size of the second dimension (index 1)
print(x.size(1)) # Outputs: 3
print(x.shape) # Outputs: torch.Size([2, 3, 4])
# Accessing a specific dimension using indexing
print(x.shape[1]) # Outputs: 3


# moving to device
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
x_data = x_data.to(device)


# Triangular parts of matrix -  Masking related functions 
x = torch.randn(4, 4)
# Returns the upper triangular part of the matrix
upper_triangle = torch.triu(x)
# Returns the lower triangular part of the matrix
lower_triangle = torch.tril(x)


# 1D tensor with a sequence of numbers, similar to Python's range()
# generate indices or a sequence of values for embedding layers or positional encodings
# Creates a tensor from 0 to 9
sequence_tensor = torch.arange(10)
# Creates a tensor from 5 to 10 with a step of 2
stepped_tensor = torch.arange(5, 11, 2)


# full 
# explicit alternative to creating a tensor with zeros or ones() 
# and then multiplying it by a scalar. 
# Useful for creating padding masks or bias tensors.
# Create a 2x3 tensor filled with the value 7.0
x = torch.full((2, 3), 7.0)
print(x)
# Output:
# tensor([[7., 7., 7.],
#         [7., 7., 7.]])


# Compute log-softmax
logits = torch.randn(4, 4)
log_probs = torch.log_softmax(logits, dim=-1)  # Shape: (3, 4)


## One-hot vectors
targets = torch.tensor([1, 2, 3])
# Step 2: One-hot encode targets
one_hot = torch.nn.functional.one_hot(targets, num_classes=logits.size(-1)).float()
# Or 
batch_size = targets.size(0)
num_classes = logits.size(-1)
# Start with a zero matrix
one_hot = torch.zeros(batch_size, num_classes, dtype=torch.float)
# Fill 1 at the target class positions
one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
print(one_hot)


## argsort, max, min
# commonly applied to the output of the final layer
losses = torch.tensor([0.5, 0.1, 0.9, 0.3])
# Get indices that would sort the tensor in ascending order
sorted_indices = torch.argsort(losses)
print(sorted_indices)
# Output: tensor([1, 3, 0, 2])

# Probabilities for 3 classes for a batch of 2
predictions = torch.tensor([[0.2, 0.7, 0.1],
                            [0.8, 0.1, 0.1]])

# Get the index of the maximum value along the last dimension
predicted_classes = torch.argmax(predictions, dim=-1)
print(predicted_classes)



## Gather function 
# Common in RL, Decoding, implementing sparse layers
# For extracting specific elements from a tensor based on another tensor of indices
# A tensor of data
data = torch.randn(3, 4)
# A tensor of indices to gather from `data`
indices = torch.tensor([[0, 1], [2, 0], [1, 3]])
# Gathers elements along dimension 1
gathered_elements = torch.gather(data, dim=-1, index=indices)


## Scatter function
# Useful for loss functions, masking
# The tensor to scatter into
target = torch.zeros(3, 4)
# The values to scatter
source = torch.ones(3, 4)
# The indices where the values will be placed
indices = torch.tensor([[0, 1], [2, 0], [1, 3]])
# Scatters the values from `source` into `target`
scattered_tensor = torch.scatter(target, dim=1, index=indices, src=source)


## Where function
# Create a tensor with 1s where x > 0 and 0s otherwise
positive_mask = torch.where(x > 0, 1, 0)
# Replace all negative values in x with 0
positive_tensor = torch.where(x > 0, x, torch.zeros_like(x))


## Concat and stack
# cat concatenates tensors along an existing dimension, while stack concatenates them along a new dimension
a = torch.randn(2, 3)
b = torch.randn(2, 3)
# Stacking creates a new dimension
stacked_tensor = torch.stack([a, b], dim=0) # Shape: (2, 2, 3)
# Concatenating combines along an existing dimension
concatenated_tensor = torch.cat([a, b], dim=0) # Shape: (4, 3)


## Einsum
# Matrix multiplication: `C[i,k] = sum(A[i,j] * B[j,k])`
A = torch.randn(3, 4)
B = torch.randn(4, 5)
matrix_mult = torch.einsum('ij,jk->ik', A, B)

# Using matmul
result = torch.matmul(a, b)
print("Resulting shape:", result.shape) # Resulting shape: torch.Size([3, 5)


## Expand
# A 1D attention mask for a batch of sequences
attention_mask = torch.tensor([1, 1, 0, 0]) # 1s for real tokens, 0s for padding
# Expand the mask to be applied to a matrix of attention scores
# Unsqueeze to add a batch and a sequence dimension
expanded_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(1, 4, 4)

# Expand vs repeat
a = torch.tensor([1, 2, 3])
# Expand: Creates a view
b = a.expand(2, 3) # Shape: (2, 3)
# Any change to 'a' will affect 'b'
a[0] = 9
print(b) # Outputs tensor([[9, 2, 3], [9, 2, 3]])
# Repeat: Creates a copy
c = a.repeat(2, 1) # Shape: (2, 3)
# 'c' is independent of 'a'
a[0] = 1 # Change back
print(c) # Outputs tensor([[9, 2, 3], [1, 2, 3]])


