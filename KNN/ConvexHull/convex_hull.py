import numpy as np
from scipy.spatial import ConvexHull
from shapely.errors import GEOSException
from shapely.geometry import Polygon


def objective_function_convex_hull_intersection_area(
    K_matrix: np.ndarray, y: np.ndarray
) -> float:
    """
    Objective function for convex hull intersection area optimization.

    This function computes the area of the intersection of the convex hulls for each class
    in the similarity space defined by the kernel matrix. The similarity space is a 2D
    projection where each point's coordinates are the sum of its similarities to all points
    in each of the two classes.

    Parameters:
    - K_matrix: np.ndarray, shape (n_samples, n_samples)
        The kernel matrix representing the similarity space for a given 'h'.
    - y: np.ndarray, shape (n_samples,)
        The binary class labels (0 or 1).

    Returns:
    - float: The area of intersection between the two class convex hulls.
    """
    try:
        # Project the samples into the 2D similarity space
        Q0 = np.sum(K_matrix[:, y == 0], axis=1)
        Q1 = np.sum(K_matrix[:, y == 1], axis=1)

        # Get the coordinates for each class in the similarity space
        C0 = np.column_stack((Q0[y == 0], Q1[y == 0]))
        C1 = np.column_stack((Q0[y == 1], Q1[y == 1]))

        # A convex hull requires at least 3 points to form an area.
        if C0.shape[0] < 3 or C1.shape[0] < 3:
            return 0.0

        # Compute the convex hull for each set of class points
        hull0 = ConvexHull(C0)
        hull1 = ConvexHull(C1)

        # Create Shapely Polygon objects from the hull vertices
        poly0 = Polygon(C0[hull0.vertices])
        poly1 = Polygon(C1[hull1.vertices])

        # Ensure polygons are valid before intersection
        if not poly0.is_valid:
            poly0 = poly0.buffer(0)
        if not poly1.is_valid:
            poly1 = poly1.buffer(0)

        # Calculate the area of the intersection of the two polygons
        intersection_area = poly0.intersection(poly1).area

        return float(intersection_area)

    except (GEOSException, Exception):
        # Fallback if any geometric or computational error occurs (e.g., collinear points)
        return 0.0
