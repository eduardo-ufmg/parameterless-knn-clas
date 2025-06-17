import numpy as np


def sparse_kernel_matrix(
    X1: np.ndarray, X2: np.ndarray | None = None, h: float = 1.0, k: int = 10
) -> np.ndarray:
    """
    Computes a sparse Gaussian RBF kernel matrix for k-nearest neighbors.

    This function calculates the kernel matrix using the Gaussian (RBF) kernel,
    but only for the 'k' nearest neighbors for each sample in X1. If a sample
    in X2 is not one of the 'k' nearest neighbors of a sample in X1, the
    corresponding kernel value is set to zero.

    The formula is:
    K(x, y) = exp(-gamma * ||x - y||^2) if y is a k-neighbor of x, else 0
    where gamma = 1 / (2 * h^2), and 'h' is the bandwidth parameter.

    Efficiency is achieved by using `np.partition` to find neighbor distances
    without performing a full sort.

    Parameters:
    - X1: np.ndarray, shape (n_samples1, n_features)
        The first set of input data points (row samples).
    - X2: np.ndarray, shape (n_samples2, n_features), optional
        The second set of input data points (column samples). If None, X2 is set to X1.
    - h: float, default=1.0
        The bandwidth parameter for the kernel. Must be a positive value.
    - k: int, default=10
        The number of nearest neighbors to consider for each sample in X1.
        Must be a positive integer. If k is greater than the number of
        samples in X2, it will be capped at the number of samples in X2.

    Returns:
    - K: np.ndarray, shape (n_samples1, n_samples2)
        The computed sparse RBF kernel matrix.

    Raises:
    - ValueError: If h is not positive, k is not positive, or if X1 and X2 have
                  mismatched feature dimensions.
    """
    # --- 1. Input Validation ---
    if h <= 0:
        raise ValueError("Bandwidth 'h' must be positive.")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("Neighbor count 'k' must be a positive integer.")
    if X2 is None:
        X2 = X1

    n_samples1, n_features1 = X1.shape
    n_samples2, n_features2 = X2.shape

    if n_features1 != n_features2:
        raise ValueError(
            f"X1 and X2 must have the same number of features, "
            f"but got {n_features1} and {n_features2}."
        )

    # Clamp k to be at most the number of available samples in X2
    k = min(k, n_samples2)

    # --- 2. Calculate Squared Euclidean Distances ---
    sq_distances = (
        np.sum(X1**2, axis=1)[:, np.newaxis]
        + np.sum(X2**2, axis=1)
        - 2 * np.dot(X1, X2.T)
    )
    # Ensure non-negativity due to potential floating-point errors
    sq_distances = np.maximum(sq_distances, 0)

    # --- 3. Identify k-Nearest Neighbors ---
    # If k equals the total number of samples, all are neighbors.
    if k == n_samples2:
        mask = np.full(sq_distances.shape, True)
    else:
        # Use np.partition to find the k-th smallest distance for each row.
        # This is more efficient than a full sort. We use k-1 because arrays
        # are 0-indexed, so the k-th element is at index k-1.
        # The result is shaped to (n_samples1, 1) for broadcasting.
        kth_smallest_sq_dist = np.partition(sq_distances, k - 1, axis=1)[
            :, k - 1, np.newaxis
        ]

        # Create a boolean mask. The comparison is broadcast across each row.
        # This mask is True for any element whose distance is <= the k-th smallest.
        # Note: In case of ties for the k-th distance, this may result in
        # more than k neighbors being included.
        mask = sq_distances <= kth_smallest_sq_dist

    # --- 4. Compute Sparse Kernel Matrix ---
    # Initialize the kernel matrix with zeros
    K = np.zeros_like(sq_distances)

    # Calculate gamma for the RBF kernel formula
    gamma = 1 / (2 * h**2)

    # Use the mask to calculate the exponential only for the k-nearest neighbors
    # This is highly efficient as it operates only on the selected elements.
    K[mask] = np.exp(-gamma * sq_distances[mask])

    return K
