import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.models.random_forest import RandomForest
import numpy as np
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split


def run_classification_example():
    print("Starting Classification Example...")  # Debug print

    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    n_trees = 10
    n_samples = len(X_train)
    B = [
        np.random.choice(n_samples, int(n_samples * 0.7), replace=True)
        for _ in range(n_trees)
    ]

    print("Training Random Forest...")  # Debug print
    rf = RandomForest(X_train, y_train, B, height=3, task="classification")

    print("Making predictions...")  # Debug print
    predictions = rf.predict(X_test)
    error_rate = rf.get_error(X_test, y_test)

    print("\nClassification Example Results:")
    print(f"Dataset size: {len(X_train)} train, {len(X_test)} test")
    print(f"Number of trees: {n_trees}")
    print(f"Error Rate: {error_rate:.4f}")

    print("\nSample predictions (first 10):")
    for i in range(min(10, len(y_test))):
        print(f"True: {y_test[i]}, Predicted: {predictions[i]}")


def run_regression_example():
    print("\nStarting Regression Example...")  # Debug print

    california = fetch_california_housing()
    X, y = california.data[:1000], california.target[:1000]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    n_trees = 10
    n_samples = len(X_train)
    B = [
        np.random.choice(n_samples, n_samples // 2, replace=True)
        for _ in range(n_trees)
    ]

    print("Training Random Forest...")  # Debug print
    rf = RandomForest(X_train, y_train, B, height=3, task="regression")

    print("Making predictions...")  # Debug print
    predictions = rf.predict(X_test)
    mse = rf.get_error(X_test, y_test)

    print("\nRegression Example Results:")
    print(f"Dataset size: {len(X_train)} train, {len(X_test)} test")
    print(f"Number of trees: {n_trees}")
    print(f"MSE: {mse:.4f}")


if __name__ == "__main__":
    print("Starting examples...")  # Debug print
    run_classification_example()
    run_regression_example()
