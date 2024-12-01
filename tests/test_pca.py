import numpy as np
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.models.pca import PCA


@pytest.fixture
def sample_data():
    """Create a simple synthetic dataset with known structure"""
    # Create data with clear principal components
    np.random.seed(42)
    n_samples = 100

    # Create data that mainly varies along [2, 1] direction
    x = np.random.normal(0, 1, n_samples)
    data = np.array([[2 * xi, xi] for xi in x])
    data += np.random.normal(0, 0.1, (n_samples, 2))

    return data


def test_pca_fit(sample_data):
    """Test that PCA correctly identifies the principal components"""
    pca = PCA()
    X_centered, _ = pca.subtract_mean(sample_data)
    U, S = pca.fit(X_centered)

    # First principal component should be approximately [2, 1] normalized
    expected_direction = np.array([2, 1]) / np.sqrt(5)
    actual_direction = U[:, 0]

    # Check if the direction is correct (or its negative, both are valid)
    aligned = np.allclose(abs(expected_direction), abs(actual_direction), atol=0.1)
    assert aligned, "First principal component doesn't match expected direction"

    # Check if eigenvalues are in descending order
    assert np.all(S[:-1] >= S[1:]), "Eigenvalues are not in descending order"


def test_reconstruction(sample_data):
    """Test that data can be projected and reconstructed with small error"""
    pca = PCA()
    X_centered, mu = pca.subtract_mean(sample_data)
    U, _ = pca.fit(X_centered)

    # Project to 1D and reconstruct
    k = 1
    Z = pca.project_data(X_centered, U, k)
    X_reconstructed = pca.recover_data(Z, U, k, mu)

    # Check reconstruction error is reasonable
    reconstruction_error = np.mean((sample_data - X_reconstructed) ** 2)
    assert reconstruction_error < 0.2, "Reconstruction error is too large"


def test_variance_explanation(sample_data):
    """Test that first component explains majority of variance"""
    pca = PCA()
    X_centered, _ = pca.subtract_mean(sample_data)
    _, S = pca.fit(X_centered)

    # Calculate variance explained by first component
    variance_ratio = S[0] / np.sum(S)

    # First component should explain most of variance (>80% for this synthetic data)
    assert variance_ratio > 0.8, "First component explains too little variance"
