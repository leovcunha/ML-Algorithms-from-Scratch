import os
import sys
import pytest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.models.knn import KNN


@pytest.fixture
def sample_data():
    """Fixture to create a simple dataset for testing"""
    X = np.array([[1, 1], [1, 2], [2, 2], [5, 5], [5, 6], [6, 6]])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


@pytest.fixture
def knn_model():
    """Fixture to create a KNN instance"""
    return KNN(n_neighbors=3)


def test_knn_fit(knn_model, sample_data):
    """Test fitting the model with sample data"""
    X, y = sample_data
    knn_model.fit(X, y)

    assert hasattr(knn_model, "X_train")
    assert hasattr(knn_model, "y_train")
    assert np.array_equal(knn_model.X_train, X)
    assert np.array_equal(knn_model.y_train, y)


def test_knn_predict(knn_model, sample_data):
    """Test predictions"""
    X, y = sample_data
    knn_model.fit(X, y)

    # Test predictions for points close to each class
    test_points = np.array(
        [[1.5, 1.5], [5.5, 5.5]]  # Should predict class 0  # Should predict class 1
    )

    predictions = knn_model.predict(test_points)
    assert predictions[0] == 0
    assert predictions[1] == 1


def test_knn_score(knn_model, sample_data):
    """Test accuracy scoring"""
    X, y = sample_data
    knn_model.fit(X, y)

    # Test on training data (should be high accuracy)
    score = knn_model.score(X, y)
    assert 0 <= score <= 1
    assert score > 0.8  # Should have good accuracy on training data


def test_euclidean_distance(knn_model):
    """Test distance calculation"""
    point1 = np.array([[0, 0]])
    point2 = np.array([3, 4])

    distance = knn_model.euclidean_distance(point1, point2)
    assert distance[0] == 5.0  # 3-4-5 triangle


def test_different_k_values(sample_data):
    """Test model with different k values"""
    X, y = sample_data

    for k in [1, 3, 5]:
        knn = KNN(n_neighbors=k)
        knn.fit(X, y)
        score = knn.score(X, y)
        assert 0 <= score <= 1
