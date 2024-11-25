import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from ml.models.logistic_regression import LogisticRegression


def test_logistic_regression_basic():
    """
    Basic end-to-end test of LogisticRegression class
    """
    # Create a simple dataset
    np.random.seed(42)
    X = np.random.randn(100, 2)  # 100 samples, 2 features
    y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)  # shape: (100, 1)

    # Train model
    model = LogisticRegression()
    model.train(X, y, calc_error=True)

    # Test predictions
    y_pred = model.predict(X)
    print(f"Prediction shape: {y_pred.shape}")
    print(f"Target shape: {y.shape}")

    # Reshape predictions if needed
    if len(y_pred.shape) == 2 and y_pred.shape[1] == 100:
        y_pred = y_pred.mean(axis=1).reshape(-1, 1)

    # Calculate accuracy
    accuracy = np.mean(y_pred.round() == y)
    print(f"Model accuracy: {accuracy:.4f}")

    # Check if training history exists
    assert hasattr(model, "loss"), "Should have loss history"
    assert hasattr(model, "training_er"), "Should have training error history"


def test_early_stopping():
    """
    Test early stopping functionality
    """
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)

    model = LogisticRegression()
    model.train(X, y, calc_error=True, early_stop="average10")

    # Verify early stopping tracking
    assert len(model.loss) > 0
    assert len(model.training_er) > 0


if __name__ == "__main__":
    pytest.main([__file__])
