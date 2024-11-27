import numpy as np
from typing import List, Optional, Union, Literal
from .decision_tree import DecisionTree


class RandomForest:
    """Random Forest supporting both regression and classification"""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        B: List[List[int]],
        height: int = 3,
        task: Literal["regression", "classification"] = "regression",
    ):
        """
        Args:
            x: Feature array
            y: Target array
            B: List of bootstrap sample indices
            height: Maximum tree height
            task: 'regression' or 'classification'
        """

        if task not in ["regression", "classification"]:
            raise ValueError("Task must be either 'regression' or 'classification'")
        if not B:  # Check if B is empty
            raise ValueError("Bootstrap samples list cannot be empty")
        if height <= 0:
            raise ValueError("Tree height must be positive")

        # ... rest of the initialization code
        self.num_trees = len(B)
        self.rforest = []
        self.task = task
        self.x_labeled = np.column_stack((x, y.reshape(-1, 1)))

        # Build forest
        for indices in B:
            self.rforest.append(DecisionTree(self.x_labeled, indices, height, task))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict values for multiple samples"""
        predictions = np.zeros(len(x))

        for i in range(len(x)):
            # Get predictions from all trees
            tree_predictions = [tree.predict(x[i]) for tree in self.rforest]

            if self.task == "regression":
                # Average for regression
                predictions[i] = np.mean(tree_predictions)
            else:
                # Majority vote for classification
                values, counts = np.unique(tree_predictions, return_counts=True)
                predictions[i] = values[np.argmax(counts)]

        return predictions

    def get_error(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate error based on task type"""
        y_pred = self.predict(x)

        if self.task == "regression":
            # MSE for regression
            return np.mean((y - y_pred) ** 2)
        else:
            # Error rate for classification
            return np.mean(y != y_pred)
