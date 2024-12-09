import numpy as np
from ml.models.naive_bayes import NaiveBayes
import pytest


def test_naive_bayes():
    X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    y = np.array([0, 0, 1, 1, 0, 1])

    model = NaiveBayes()
    model.fit(X, y)

    X_test = np.array([[2, 1], [6, 9]])
    y_pred = model.predict(X_test)

    # Expected predictions based on the simple dataset
    expected_predictions = np.array([0, 1])

    np.testing.assert_array_equal(y_pred, expected_predictions)
