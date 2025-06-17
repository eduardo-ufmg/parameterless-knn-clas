import numpy as np
from scipy.spatial.distance import cdist, pdist


def intra_class_average_distance(C: np.ndarray) -> float:
    """
    Computes the average distance between points in the same class.

    Parameters:
    - C: np.ndarray, shape (n_samples, 2)
        The samples from the similarity space for a specific class.

    Returns:
    - float: The average distance between points in the same class.
    """
    # If there are fewer than 2 points, the distance is 0
    if C.shape[0] < 2:
        return 0.0
    # pdist computes the condensed distance matrix (1D array of pairwise distances)
    distances = pdist(C, "euclidean")
    # The average distance is the mean of these pairwise distances
    return float(np.mean(distances))


def inter_class_average_distance(C0: np.ndarray, C1: np.ndarray) -> float:
    """
    Computes the average distance between points in different classes.

    Parameters:
    - C0: np.ndarray, shape (n_samples_class_0, 2)
        The samples from the similarity space for class 0.
    - C1: np.ndarray, shape (n_samples_class_1, 2)
        The samples from the similarity space for class 1.

    Returns:
    - float: The average distance between points in different classes.
    """
    # If either class has no points, the distance is 0
    if C0.shape[0] == 0 or C1.shape[0] == 0:
        return 0.0
    # cdist computes the distance between each pair of the two collections of inputs
    dist_matrix = cdist(C0, C1, "euclidean")
    # The average is the mean of all distances in the resulting matrix
    return float(np.mean(dist_matrix))


def objective_function_spatial_spread(K_matrix: np.ndarray, y: np.ndarray) -> float:
    """
    Objective function for spatial spread optimization.
    This function computes the spatial spread of the similarity space defined by the kernel matrix K_matrix.
    The spatial spread is defined as the average distance between points in the same class,
    minus the absolute difference between the average distance within classes,
    plus the average distance between points in different classes.

    Parameters:
    - K_matrix: np.ndarray, shape (n_samples, n_samples)
        The kernel matrix representing the similarity space.
    - y: np.ndarray, shape (n_samples,)
        The class labels for the samples.

    Returns:
    - float: The computed spatial spread value.
    """

    # Projects the samples into the similarity space
    Q0 = np.sum(K_matrix[:, y == 0], axis=1)
    Q1 = np.sum(K_matrix[:, y == 1], axis=1)

    # Extracts the samples from the similarity space for each class
    C0 = np.column_stack((Q0[y == 0], Q1[y == 0]))
    C1 = np.column_stack((Q0[y == 1], Q1[y == 1]))

    # Computes the average distance between points in the same class
    intra_class_average_distance_0 = intra_class_average_distance(C0)
    intra_class_average_distance_1 = intra_class_average_distance(C1)

    intra_class_average_distance_mean = (
        intra_class_average_distance_0 + intra_class_average_distance_1
    ) / 2

    intra_class_average_distance_abs_diff = abs(
        intra_class_average_distance_0 - intra_class_average_distance_1
    )

    inter_class_average_distance_value = inter_class_average_distance(C0, C1)

    # Computes the spatial spread value
    spatial_spread_value = (
        intra_class_average_distance_mean
        - intra_class_average_distance_abs_diff
        + inter_class_average_distance_value
    )

    return spatial_spread_value
