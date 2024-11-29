# src/models/hierarchical_clustering.py

import numpy as np
import math
from typing import List, Tuple


class Node:
    """
    Node in hierarchical clustering tree.
    Each node represents either a single observation or merged clusters.
    """

    def __init__(self, observations: tuple, height: float = 0, left=None, right=None):
        if len(observations) == 1:  # Leaf node (single observation)
            self.is_leaf = True
            self.height = 0
            self.observations = observations
            self.left = None
            self.right = None
        else:  # Internal node (merged clusters)
            self.is_leaf = False
            self.height = height
            self.observations = observations
            self.left = left
            self.right = right

    def get_obs(self):
        """Return observations indices in this cluster"""
        return self.observations


def euclidian_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points"""
    return np.sqrt(np.sum((x1 - x2) ** 2))


def single_linkage(c1: np.ndarray, c2: np.ndarray) -> float:
    """Shortest distance between elements of two clusters"""
    d = [euclidian_distance(elem1, elem2) for elem1 in c1 for elem2 in c2]
    return min(d)


def complete_linkage(c1: np.ndarray, c2: np.ndarray) -> float:
    """Longest distance between elements of two clusters"""
    d = [euclidian_distance(elem1, elem2) for elem1 in c1 for elem2 in c2]
    return max(d)


def average_linkage(c1: np.ndarray, c2: np.ndarray) -> float:
    """Average distance between elements of two clusters"""
    d = [euclidian_distance(elem1, elem2) for elem1 in c1 for elem2 in c2]
    return np.mean(d)


def centroid_linkage(c1: np.ndarray, c2: np.ndarray) -> float:
    """Distance between centroids of two clusters"""
    return euclidian_distance(np.mean(c1, axis=0), np.mean(c2, axis=0))


class AgglomerativeClustering:
    """
    Bottom-up hierarchical clustering implementation.
    Starts with each point as its own cluster and iteratively
    merges the most similar clusters based on chosen linkage criterion.
    """

    def __init__(self, data, linkage=single_linkage):  # Default to single linkage
        self.data = data
        self.clusters = {}  # Dictionary to store clusters at each level
        # Initialize level 0: each point is its own cluster
        self.clusters[0] = [Node((idx,)) for idx, _ in enumerate(data)]

        i = 0
        # Continue merging until only one cluster remains
        while len(self.clusters[i]) > 1:
            # Find the two most similar clusters to merge
            c1, c2, distance = self.get_most_similar(self.clusters[i], linkage)

            # Create new level i+1
            # First, copy all clusters except the two being merged
            self.clusters[i + 1] = [c for c in self.clusters[i] if c != c1 and c != c2]

            # Create new merged cluster and add it to level i+1
            self.clusters[i + 1].append(
                Node(
                    observations=c1.get_obs() + c2.get_obs(),  # Combine observations
                    left=c1,  # Store merged clusters as children
                    right=c2,
                    height=distance,  # Store merge height/distance
                )
            )
            i += 1
        self.levels = i  # Store total number of merge steps

    def get_most_similar(self, clusters, linkage):
        """
        Find the two closest clusters to merge based on linkage criterion.

        Args:
            clusters: List of current cluster nodes
            linkage: Function to calculate distance between clusters

        Returns:
            Tuple of (cluster1, cluster2, distance) for the closest pair
        """
        min_dist = math.inf
        argmin = ()
        # Compare each pair of clusters
        for i, c1 in enumerate(clusters):
            for j, c2 in enumerate(clusters):
                # Calculate distance between clusters using linkage function
                dist = linkage(self.data[c1.get_obs(), :], self.data[c2.get_obs(), :])
                # Update minimum if this pair is closer
                if i != j and dist < min_dist:
                    min_dist = dist
                    argmin = (c1, c2)
        return argmin[0], argmin[1], min_dist

    def cluster_vals(self, i):
        """Get the observations in each cluster at level i"""
        return [node.get_obs() for node in self.clusters[i]]

    def dendogram(self):
        """Return the full hierarchical clustering structure"""
        return self.clusters

    def get_clusters(self, K):
        """
        Extract K clusters from the dendrogram by finding the appropriate level.

        Args:
            K: Desired number of clusters

        Returns:
            List of cluster nodes at the level with K clusters,
            or empty list if K clusters cannot be found
        """
        n = len(self.clusters.keys())  # Total number of levels
        # Search from bottom up for level with K clusters
        for i in range(n):
            if len(self.clusters[n - 1 - i]) == K:
                return self.clusters[n - 1 - i]
        return []  # Return empty list if K clusters not found
