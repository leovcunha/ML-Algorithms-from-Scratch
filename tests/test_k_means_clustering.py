import numpy as np
import pytest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.models.k_means_clustering import k_means_clustering


def test_k_means_clustering():
    X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k = 2
    labels, centroids = k_means_clustering(X, k)

    # Check if the number of unique labels is equal to k
    assert len(np.unique(labels)) == k

    # Check if the centroids are not empty
    assert centroids.shape[0] == k


if __name__ == "__main__":
    test_k_means_clustering()
