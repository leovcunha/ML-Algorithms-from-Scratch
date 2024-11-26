import numpy as np
from typing import List
from .decision_tree import DecisionTree


class RandomForest:
    """
    Random Forest implementation for regression.
    Combines multiple decision trees using bootstrap sampling and feature subsampling.
    """

    def __init__(
        self, x: np.ndarray, y: np.ndarray, B: List[List[int]], height: int = 3
    ):
        """
        Initialize and build the random forest.

        Args:
            x: Feature array of shape (n_samples, n_features)
            y: Target array of shape (n_samples,)
            B: List of bootstrap sample indices for each tree
            height: Maximum height allowed for each tree
        """
        # Store number of trees
        self.num_trees = len(B)
        # Initialize list to store trees
        self.rforest = []
        # Combine features and target for decision tree implementation
        self.x_labeled = np.column_stack((x, y.reshape(-1, 1)))

        # Build each tree using different bootstrap samples
        for indices in B:
            self.rforest.append(DecisionTree(self.x_labeled, indices, height))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict target values for multiple samples.
        Prediction is average of all tree predictions.

        Args:
            x: Samples to predict, shape (n_samples, n_features)
        Returns:
            Predictions array of shape (n_samples,)
        """
        predictions = np.zeros(len(x))
        # For each sample
        for i in range(len(x)):
            # Average predictions from all trees
            predictions[i] = np.mean([tree.predict(x[i]) for tree in self.rforest])
        return predictions

    def get_mse(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate Mean Squared Error between predictions and true values.

        Args:
            x: Feature array
            y: True target values
        Returns:
            MSE value
        """
        y_pred = self.predict(x)
        return np.mean((y - y_pred) ** 2)
