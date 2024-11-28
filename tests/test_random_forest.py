# tests/test_random_forest.py
import numpy as np
import pytest
from ml.models.random_forest import RandomForest


@pytest.fixture
def setup_regression_data():
    """Fixture for regression test data"""
    np.random.seed(42)
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])  # y = 2x
    n_trees = 3
    B = [np.array([0, 1, 2, 2, 4]) for _ in range(n_trees)]
    return X, y, B


@pytest.fixture
def setup_classification_data():
    """Fixture for classification test data"""
    np.random.seed(42)
    X = np.array([[1], [2], [3], [4], [5], [1], [2], [3], [4], [5]])
    y = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1])
    n_trees = 3
    B = [np.array([0, 1, 2, 2, 4, 5, 6, 7, 8, 9]) for _ in range(n_trees)]
    return X, y, B


def test_regression_initialization(setup_regression_data):
    """Test Random Forest initialization for regression"""
    X, y, B = setup_regression_data
    rf = RandomForest(X, y, B, height=2, task="regression")
    assert len(rf.rforest) == len(B)
    assert rf.task == "regression"


def test_classification_initialization(setup_classification_data):
    """Test Random Forest initialization for classification"""
    X, y, B = setup_classification_data
    rf = RandomForest(X, y, B, height=2, task="classification")
    assert len(rf.rforest) == len(B)
    assert rf.task == "classification"


def test_regression_prediction(setup_regression_data):
    """Test regression predictions"""
    X, y, B = setup_regression_data
    rf = RandomForest(X, y, B, height=2, task="regression")
    predictions = rf.predict(np.array([[2.5]]))
    # Should predict close to y = 2x (which would be 5.0)
    assert 4 <= predictions[0] <= 6  # Changed to include boundary values


def test_classification_prediction(setup_classification_data):
    """Test classification predictions"""
    X, y, B = setup_classification_data
    rf = RandomForest(X, y, B, height=2, task="classification")
    predictions = rf.predict(np.array([[1.5], [4.5]]))
    assert predictions[0] == 0  # Should predict class 0
    assert predictions[1] == 1  # Should predict class 1


def test_regression_error(setup_regression_data):
    """Test regression error calculation"""
    X, y, B = setup_regression_data
    rf = RandomForest(X, y, B, height=2, task="regression")
    error = rf.get_error(X, y)
    assert error >= 0  # MSE should be non-negative


def test_classification_error(setup_classification_data):
    """Test classification error calculation"""
    X, y, B = setup_classification_data
    rf = RandomForest(X, y, B, height=2, task="classification")
    error = rf.get_error(X, y)
    assert 0 <= error <= 1  # Error rate should be between 0 and 1


def test_invalid_task(setup_regression_data):
    """Test if invalid task raises ValueError"""
    X, y, B = setup_regression_data
    with pytest.raises(ValueError):
        RandomForest(X, y, B, height=2, task="invalid")


def test_empty_bootstrap_samples(setup_regression_data):
    """Test handling of empty bootstrap samples"""
    X, y, _ = setup_regression_data
    empty_B = []
    with pytest.raises(ValueError):
        RandomForest(X, y, empty_B, height=2)


def test_height_validation(setup_regression_data):
    """Test validation of tree height parameter"""
    X, y, B = setup_regression_data
    with pytest.raises(ValueError):
        RandomForest(X, y, B, height=0)  # Height should be positive


def test_data_consistency(setup_regression_data):
    """Test if data and bootstrap indices are consistent"""
    X, y, B = setup_regression_data
    # Modify B to include invalid index
    invalid_B = [np.array([0, 1, 2, 5, 6]) for _ in range(3)]  # Index 6 is invalid
    with pytest.raises(IndexError):
        RandomForest(X, y, invalid_B, height=2)
