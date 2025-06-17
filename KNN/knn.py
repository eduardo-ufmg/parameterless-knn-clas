import sys
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from scipy.stats import mode
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

sys.path.append(str(Path(__file__).parent))

from ConvexHull.convex_hull import \
    objective_function_convex_hull_intersection_area
from Dissimilarity.dissimilarity import objective_function_dissimilarity
from Kernel.kernel import sparse_kernel_matrix
from SpatialSpread.spatial_spread import objective_function_spatial_spread


def gabriel_graph(X: np.ndarray) -> nx.Graph:
    """
    Computes the Gabriel Graph for a set of N-dimensional points.

    This implementation is efficient because it relies on the principle that the
    Gabriel Graph is a subgraph of the Delaunay Triangulation. It first builds
    the Delaunay graph and then prunes the edges that do not meet the Gabriel
    criterion.

    An edge (u, v) is a Gabriel edge if its diametric sphere is empty of any
    other points from the set. For an edge (u, v) forming a triangle with a
    third point w, this condition is met if the angle at w is not obtuse.

    Parameters
    ----------
    X : np.ndarray
        A numpy array of shape (n_samples, n_features) representing the point
        coordinates.

    Returns
    -------
    nx.Graph
        A networkx Graph object representing the Gabriel Graph. The nodes are
        integers corresponding to the indices of the points in the input array X.
    """
    # For Delaunay triangulation in d-dimensions, at least d+1 points are required.
    # If not enough points, return a complete graph as a sensible default.
    if X.shape[0] <= X.shape[1]:
        return nx.complete_graph(X.shape[0])

    # 1. Compute the Delaunay triangulation. This is much faster than checking all
    #    pairs of points.
    try:
        tri = Delaunay(X)
    except Exception:
        # Fallback for degenerate cases (e.g., all points are collinear).
        return nx.complete_graph(X.shape[0])

    # 2. Build a map from each vertex to the list of simplices (triangles) it's part of.
    #    This allows for efficient lookup of neighboring triangles.
    n_samples = X.shape[0]
    vertex_to_simplices = [[] for _ in range(n_samples)]
    for i, simplex in enumerate(tri.simplices):
        for vertex in simplex:
            vertex_to_simplices[vertex].append(i)

    # 3. Initialize the graph and iterate through unique Delaunay edges to check
    #    the Gabriel condition.
    G = nx.Graph()
    G.add_nodes_from(range(n_samples))

    # Using a set ensures we only check each edge once.
    checked_edges = set()

    for i, simplex in enumerate(tri.simplices):
        for u, v in [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])]:
            # Ensure u < v to have a canonical representation for the edge.
            if u > v:
                u, v = v, u
            if (u, v) in checked_edges:
                continue
            checked_edges.add((u, v))

            # An edge is presumed to be a Gabriel edge until a counterexample is found.
            is_gabriel_edge = True

            # Find common simplices between vertices u and v to identify the third
            # vertex 'w' of adjacent triangles.
            common_simplices_indices = set(vertex_to_simplices[u]).intersection(vertex_to_simplices[v])

            for simplex_idx in common_simplices_indices:
                # Find the third vertex 'w' in the triangle.
                third_vertex = list(set(tri.simplices[simplex_idx]) - {u, v})[0]
                w = third_vertex

                # Check the Gabriel condition: angle 'uwv' must not be obtuse.
                # This is true if: d(u,w)² + d(v,w)² >= d(u,v)²
                d_uv_sq = np.sum((X[u] - X[v])**2)
                d_uw_sq = np.sum((X[u] - X[w])**2)
                d_vw_sq = np.sum((X[v] - X[w])**2)

                if d_uw_sq + d_vw_sq < d_uv_sq:
                    is_gabriel_edge = False
                    break  # Found a violating point; this is not a Gabriel edge.

            if is_gabriel_edge:
                G.add_edge(u, v)

    return G


class ParameterlessKNN(BaseEstimator, ClassifierMixin):
    """
    A K-Nearest Neighbors (KNN) classifier that automatically optimizes its
    hyperparameters (kernel bandwidth 'h' and number of neighbors 'k').

    This classifier can use a data reduction technique based on the Gabriel
    Graph (CLAS) and optimizes hyperparameters by maximizing a chosen objective
    function.

    Parameters
    ----------
    optimization : {'accuracy', 'dissimilarity', 'spread', 'convex_hull'}, default='accuracy'
        The objective function to maximize for hyperparameter tuning.
        - 'accuracy': Maximizes cross-validated classification accuracy.
        - 'dissimilarity': Maximizes a metric of class dissimilarity.
        - 'spread': Maximizes the spatial spread between classes.
        - 'convex_hull': Maximizes the intersection area of class convex hulls.
                        This is intentionally maximized to ensure some class overlap
                        and prevent overfitting on highly separable data.

    clas : bool, default=False
        If True, enables the CLAS data reduction method. This requires the
        `networkx` library. The training set is reduced to "support samples"
        that lie on the decision boundary.

    Attributes
    ----------
    h_ : float
        The optimal kernel bandwidth 'h' found during optimization.

    k_ : int
        The optimal number of neighbors 'k' found during optimization.

    X_ : np.ndarray
        The training data, possibly reduced by the CLAS method.

    y_ : np.ndarray
        The training labels, corresponding to X_.

    classes_ : np.ndarray
        The unique class labels found in the training data.
    """

    def __init__(self, optimization="accuracy", clas=False):
        self.optimization = optimization
        self.clas = clas

    def _get_support_samples(self, X, y):
        """
        Reduces the dataset to only support samples using the CLAS method.
        It builds a Gabriel Graph and keeps only samples connected by an
        edge to a sample of the opposite class.
        """

        G = gabriel_graph(X)

        support_indices = set()
        for u, v in G.edges():
            if y[u] != y[v]:
                support_indices.add(u)
                support_indices.add(v)

        if not support_indices:
            print("CLAS: No support edges found. Using the full dataset.")
            return X, y

        indices = sorted(list(support_indices))
        return X[indices], y[indices]

    def _objective(self, params, X, y):
        """
        The objective function to be minimized by the optimization algorithm.
        It computes the score for a given (h, k) pair and returns its negative,
        as the optimizer performs minimization.
        """
        h, k_float = params
        # Get the base k value from the optimizer
        k_base = int(np.round(k_float))

        score = 0.0

        if self.optimization == "accuracy":
            # Use 3-fold stratified cross-validation for a stable accuracy estimate.
            skf = StratifiedKFold(n_splits=3, shuffle=True)
            accuracies = []
            for train_idx, val_idx in skf.split(X, y):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                n_train_samples = X_train.shape[0]
                if n_train_samples <= 1:
                    accuracies.append(0.0)
                    continue

                # Use a local k for this fold, validated against the fold's size
                k = k_base
                if k >= n_train_samples:
                    k = n_train_samples - 1

                # Ensure k is a positive, odd integer
                if k % 2 == 0:
                    k -= 1
                k = max(1, k)

                # Temporarily fit a standard k-NN on the sub-train fold
                dist_matrix = cdist(X_val, X_train)
                
                # Find the k nearest neighbors. The k value must be less than n_train_samples.
                neighbor_indices = np.argpartition(dist_matrix, k, axis=1)[:, :k]
                neighbor_labels = y_train[neighbor_indices]
                
                # Predict using majority vote
                predictions, _ = mode(neighbor_labels, axis=1, keepdims=True)
                accuracies.append(accuracy_score(y_val, predictions))
            score = np.mean(accuracies)

        else:
            # For other metrics, validate k against the full dataset X.
            n_samples = X.shape[0]
            k = k_base

            if k >= n_samples:
                k = n_samples - 1 if n_samples > 1 else 1

            # Ensure k is a positive, odd integer
            if k % 2 == 0:
                k -= 1
            k = max(1, k)

            # Compute the sparse kernel matrix on the whole dataset
            K_matrix = sparse_kernel_matrix(X, X, h=h, k=k)

            if self.optimization == "dissimilarity":
                score = objective_function_dissimilarity(K_matrix, y)
            elif self.optimization == "spread":
                score = objective_function_spatial_spread(K_matrix, y)
            elif self.optimization == "convex_hull":
                score = objective_function_convex_hull_intersection_area(K_matrix, y)

        # We want to maximize the score, so we return its negative
        return -score

    def fit(self, X, y):
        """
        Fits the classifier to the data. This involves optional CLAS reduction
        and then running the optimization to find the best h and k.
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        if self.clas:
            X_fit, y_fit = self._get_support_samples(X, y)
        else:
            X_fit, y_fit = X, y

        # Define the search space (bounds) for the optimization
        n_samples = X_fit.shape[0]
        # Heuristic for h: based on the median of pairwise distances
        h_med = np.median(cdist(X_fit, X_fit))
        bounds = [(1e-3, h_med * 10), (1, n_samples - 1)]

        result = differential_evolution(
            self._objective,
            bounds,
            args=(X_fit, y_fit),
            updating='deferred',
            workers=-1,  # Use all available CPU cores
        )

        self.h_ = result.x[0]
        k_opt = int(np.round(result.x[1]))

        # Ensure the final k is valid for the stored training set
        n_samples = X_fit.shape[0]
        if k_opt >= n_samples:
            k_opt = n_samples - 1

        # Ensure k is a positive, odd integer
        if k_opt % 2 == 0:
            k_opt -= 1
        self.k_ = max(1, k_opt)

        # Store the data that the final classifier will use for predictions
        self.X_ = X_fit
        self.y_ = y_fit

        return self

    def predict(self, X):
        """
        Predicts class labels for new data points using the optimized k.
        """
        check_is_fitted(self)
        X = check_array(X)

        # Calculate distances from test points to the stored training points
        dist_matrix = cdist(X, self.X_)

        # Find the k_ nearest neighbor indices for each test point
        neighbor_indices = np.argpartition(dist_matrix, self.k_, axis=1)[:, : self.k_]

        # Get the labels of those neighbors
        neighbor_labels = self.y_[neighbor_indices]

        # Predict the class with the highest frequency (majority vote)
        predictions, _ = mode(neighbor_labels, axis=1, keepdims=True)

        return predictions.ravel()
