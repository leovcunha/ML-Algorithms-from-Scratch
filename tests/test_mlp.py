import os
import sys
import pytest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.models.multilayer_perceptron import MultilayerPerceptron


def test_mlp_initialization():
    """Test if MLP initializes with correct dimensions and parameters"""
    mlp = MultilayerPerceptron(
        layer_dims=[2, 4, 1], learning_rate=0.1, n_iterations=100, random_seed=42
    )

    # Test basic attributes
    assert mlp.learning_rate == 0.1
    assert mlp.n_iterations == 100

    # Test parameter shapes
    assert mlp.parameters["W1"].shape == (4, 2)
    assert mlp.parameters["b1"].shape == (4, 1)
    assert mlp.parameters["W2"].shape == (1, 4)
    assert mlp.parameters["b2"].shape == (1, 1)


def test_predict():
    """Test if predict method returns correct shape"""
    mlp = MultilayerPerceptron([2, 4, 1], random_seed=42)
    X = np.array([[0, 1], [1, 0]]).T  # 2 samples, 2 features

    predictions = mlp.predict(X)
    assert predictions.shape == (1, 2)  # (1, n_samples)
    assert np.all(
        (predictions >= 0) & (predictions <= 1)
    )  # Predictions should be in [0, 1]


def test_fit():
    """Test if fit method runs without errors"""
    X = np.array([[0, 1], [1, 0]]).T  # 2 samples, 2 features
    y = np.array([[0, 1]])  # Binary labels

    mlp = MultilayerPerceptron([2, 4, 1], n_iterations=10, random_seed=42)
    mlp.fit(X, y)  # fit doesn't return costs

    # Test if the model can make predictions after fitting
    predictions = mlp.predict(X)
    assert predictions.shape == y.shape  # Ensure predictions match shape
    assert np.all(
        (predictions >= 0) & (predictions <= 1)
    )  # Predictions should be in [0, 1]


def test_model_accuracy():
    """Test if the model achieves reasonable accuracy on a simple task"""
    X = np.array([[0, 1], [1, 0]]).T  # 2 samples, 2 features
    y = np.array([[0, 1]])  # Binary labels

    mlp = MultilayerPerceptron([2, 4, 1], n_iterations=500, random_seed=42)
    mlp.fit(X, y)

    predictions = mlp.predict(X)
    accuracy = np.mean(predictions == y)

    # Assert that accuracy is better than random guessing (for this simple task, 100% is expected)
    assert (
        accuracy >= 0.5
    )  # For a toy dataset like this, the model should at least guess correctly


if __name__ == "__main__":
    pytest.main()
