import numpy as np
import torch


def kmeans_naive(X, k, num_iters=100, seed=42):
    """
    Non-batch K-means: loops over points for cluster assignment.
    X: (n_samples, n_features)
    """
    np.random.seed(seed)
    n, d = X.shape
    # Randomly choose k points as initial centroids
    centroids = X[np.random.choice(n, k, replace=False)]
    for _ in range(num_iters):
        # Assign clusters (loop over each point)
        labels = []
        for x in X:
            dists = [np.linalg.norm(x - c) for c in centroids]
            labels.append(np.argmin(dists))
        labels = np.array(labels)
        # Update centroids
        new_centroids = []
        for i in range(k):
            points = X[labels == i]
            # print(f"points shape: {points.shape}")  # (num_points, d)
            if len(points) > 0:
                new_centroids.append(points.mean(axis=0))  # List of (d)
            else:
                new_centroids.append(centroids[i])  # keep old if empty
        new_centroids = np.vstack(new_centroids)
        # print(f"new_centroids shape: {new_centroids.shape}") # (k, d)
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return centroids, labels
