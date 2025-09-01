import torch
import numpy as np

def knn_naive(X_train, y_train, X_test, k=3):
    """
    X_train: (n_train, d)
    y_train: (n_train,)
    X_test:  (n_test, d)
    Returns predicted labels for X_test
    """
    n_test = X_test.shape[0]
    y_pred = []
    for x in X_test:
        # Compute distances to all training points
        dists = np.linalg.norm(X_train - x, axis=1)
        # SA: TODO: Check print below
        # print(f"Naive numpy dist shape: {dists.shape}. Note it is (n2,)")
        # Get indices of k nearest neighbors
        knn_idx = np.argsort(dists)[:k]
        # Majority vote
        knn_labels = y_train[knn_idx]
        counts = np.bincount(knn_labels)
        y_pred.append(np.argmax(counts))
    return np.array(y_pred)

def knn_batch(X_train, y_train, X_test, k=3):
    """
    Fully vectorized KNN using broadcasting.
    X_train: (n_train, d)
    X_test:  (n_test, d)
    """
    # Compute pairwise distances: (n_test, n_train)
    # Original x_test (n1,d) ; x_train (n2,d)
    # x_test (n1,1,d) ; x_train (1,n2,d)
    dists = np.linalg.norm(X_test[:, None, :] - X_train[None, :, :], axis=2)
    print(f"Vector numpy dist shape: {dists.shape}. Note it is still (n1,n2)")
    # This gives (n1,n2)
    # Get k nearest neighbors
    knn_idx = np.argsort(dists, axis=1)[:, :k]
    # Get majority vote for each test point
    y_pred = np.array([np.bincount(y_train[idx]).argmax() for idx in knn_idx])
    return y_pred


def knn_torch(X_train, y_train, X_test, k=3, device="cpu"):
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    # Compute distances (n_test, n_train)
    dists = torch.cdist(X_test, X_train)  # efficient GPU pairwise distance
    # TODO: or see
    # dists_manual = torch.norm(X_test[:, None, :] - X_train[None, :, :], dim=2)
    # print(dists.shape) # [10, 100]
    # Get k nearest neighbors - Largest false since min dist required
    knn_idx = torch.topk(dists, k=k, largest=False).indices  # (n_test, k)
    # Majority vote
    y_pred = []
    for idx in knn_idx:
        labels = y_train[idx]
        counts = torch.bincount(labels)
        y_pred.append(torch.argmax(counts))
    y_pred = torch.stack(y_pred) # Convert list to tensor
    return y_pred


def knn_torch_vectorized(X_train, y_train, X_test, k=3, num_classes=None, device="cpu"):
    """
    Fully vectorized KNN in PyTorch (GPU-friendly).
    X_train: (n_train, d)
    y_train: (n_train,) long tensor of class labels
    X_test:  (n_test, d)
    num_classes: total number of classes (optional, inferred if None)
    Instead of looping, we use scatter:

    one_hot is initialized as zeros: shape (n_test, k, num_classes).
    knn_labels.unsqueeze(-1) makes it (n_test, k, 1).
    scatter_(dim=2, index, src=1) places 1 (src) in the right class position for each neighbor.
    """
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    n_test = X_test.shape[0]

    # imp to create 1-hot vector
    if num_classes is None:
        num_classes = int(y_train.max().item() + 1)

    # Compute distances: (n_test, n_train)
    dists = torch.cdist(X_test, X_train)  # Euclidean distances

    # Indices of k nearest neighbors
    knn_idx = torch.topk(dists, k=k, largest=False).indices  # (n_test, k)

    # Gather neighbor labels: (n_test, k)
    knn_labels = y_train[knn_idx]
    print(f"knn_labels shape: {knn_labels.shape}")

    # One-hot encode and sum to get counts per class: (n_test, num_classes)
    one_hot = torch.zeros(n_test, k, num_classes, device=device) # (n_test, k, n_classes)
    print(f"one_hot shape: {one_hot.shape}")
    one_hot.scatter_(2, knn_labels.unsqueeze(-1), 1) # (n_test, k, n_classes); dim=2, idx=label; value=1
    class_counts = one_hot.sum(dim=1) # (n_test, n_classes)
    print(f"class_counts shape: {class_counts.shape}")

    # Predicted class: argmax per test point
    y_pred = class_counts.argmax(dim=1) # (n_test, )
    return y_pred


if __name__ == "__main__":

    # Dummy dataset
    X_train = np.random.randn(100, 2)
    y_train = np.random.randint(0, 3, size=100)
    X_test = np.random.randn(10, 2)
    y_pred_naive = knn_naive(X_train, y_train, X_test, k=5)
    y_pred_batch = knn_batch(X_train, y_train, X_test, k=5)
    print("Naive:", y_pred_naive)
    print("Vector:", y_pred_batch)
    # PyTorch version
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_pred_torch = knn_torch(X_train_t, y_train_t, X_test_t, k=5, device=device)
    print("PyTorch:", y_pred_torch)

    # Dummy dataset
    # X_train = torch.randn(1000, 16, device=device)
    # y_train = torch.randint(0, 5, (1000,), device=device)
    # X_test = torch.randn(10, 16, device=device)
    y_pred = knn_torch_vectorized(X_train, y_train, X_test, k=5, device=device)
    print("PyTorch :", y_pred)
