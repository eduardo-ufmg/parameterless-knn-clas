import numpy as np


def objective_function_dissimilarity(K_matrix: np.ndarray, y: np.ndarray) -> float:
    """
    Calculates the dissimilarity metric for kernel matrix optimization.

    This function implements the dissimilarity metric described in the paper
    "Width optimization of RBF kernels for binary classification of support
    vector machines: A density estimation-based approach" by Menezes et al. The goal
    is to find a kernel width 'h' that maximizes the separability of classes in
    the likelihood space.

    The dissimilarity is defined in Equation (13) of the paper as the product of the
    Euclidean distance and the cosine of the angle between the class similarity vectors.

    Parameters:
    - K_matrix: The pre-computed kernel matrix for a given 'h'.
    - y: The binary class labels (0 or 1) for the dataset.

    Returns:
    The calculated dissimilarity metric.
    """
    # Use boolean indexing for splitting the data by class
    class_0_mask = y == 0
    class_1_mask = y == 1

    # Extract the four sub-matrices from the main kernel matrix
    K_00 = K_matrix[class_0_mask][:, class_0_mask]
    K_01 = K_matrix[class_0_mask][:, class_1_mask]
    K_10 = K_matrix[class_1_mask][:, class_0_mask]
    K_11 = K_matrix[class_1_mask][:, class_1_mask]

    # If either class is empty, dissimilarity cannot be computed.
    if K_00.size == 0 or K_11.size == 0:
        return 0.0

    # Calculate the average similarity for each class pairing (S_ij from the paper)
    S_00 = np.mean(K_00)
    S_01 = np.mean(K_01)
    S_10 = np.mean(K_10)
    S_11 = np.mean(K_11)

    # Construct the class similarity vectors (V1 and V2 in the paper)
    # V0 represents the mean point of mapped samples for class 0
    # V1 represents the mean point of mapped samples for class 1
    V0 = np.array([S_00, S_01])
    V1 = np.array([S_10, S_11])

    # Calculate the vector norms
    norm_V0 = np.linalg.norm(V0)
    norm_V1 = np.linalg.norm(V1)

    # Avoid division by zero if a vector norm is zero
    if norm_V0 == 0 or norm_V1 == 0:
        return 0.0

    # Calculate the cosine of the angle between the class vectors
    dot_product = np.dot(V0, V1)
    cosine_angle = dot_product / (norm_V0 * norm_V1)

    # Calculate the Euclidean distance between the class vectors
    euclidean_distance = np.linalg.norm(V0 - V1)

    # The dissimilarity metric from Equation (12) in the paper
    dissimilarity = cosine_angle * euclidean_distance

    return dissimilarity
