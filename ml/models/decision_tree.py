import numpy as np
from typing import List, Optional, Union, Literal


class DecisionTree:
    """
    Decision Tree that supports both classification and regression tasks.
    For regression: Uses RSS (Residual Sum of Squares) for splitting
    For classification: Uses Gini impurity for splitting
    """

    def __init__(
        self,
        data: np.ndarray,
        indices: List[int],
        depth: int,
        task: Literal["regression", "classification"] = "regression",
    ):
        """
        Initialize and build the decision tree recursively.

        Args:
            data: Combined features and target array where last column is the target
            indices: List of indices to use from data (supports bootstrap sampling)
            depth: Maximum allowed depth of the tree
            task: Whether to perform 'regression' or 'classification'
        """
        # Initialize node attributes
        self.leaf = False  # Flag to indicate if node is a leaf
        self.prediction = None  # Store prediction value for leaf nodes
        self.attr = None  # Store splitting attribute/feature index
        self.split = None  # Store splitting value/threshold
        self.L = None  # Left subtree
        self.R = None  # Right subtree
        self.task = task  # Store task type

        # Get unique values in target variable
        unique_vals = set(data[indices, -1])

        # Case 1: If we've reached maximum depth, make this a leaf node
        if depth == 0:
            self.leaf = True
            self.prediction = self._get_prediction(data, indices)

        # Case 2: If all samples have same target value, make this a leaf node
        elif len(unique_vals) == 1:
            self.leaf = True
            self.prediction = data[indices[0], -1]

        # Case 3: Try to split the node
        else:
            tree_info = self._generate(data, indices, depth)
            # If splitting fails, make this a leaf node
            if not tree_info:
                self.leaf = True
                self.prediction = self._get_prediction(data, indices)
            # If splitting succeeds, create internal node with left and right children
            else:
                self.attr, self.split, self.L, self.R = tree_info

    def _get_prediction(
        self, data: np.ndarray, indices: List[int]
    ) -> Union[float, int]:
        """
        Get prediction based on task type.
        For regression: returns mean of target values
        For classification: returns most common class

        Args:
            data: Full dataset
            indices: Indices to consider
        Returns:
            Predicted value (float for regression, int for classification)
        """
        if self.task == "regression":
            return self._avg(data, indices)
        else:  # classification
            values, counts = np.unique(data[indices, -1], return_counts=True)
            return values[np.argmax(counts)]

    @staticmethod
    def _avg(data: np.ndarray, indices: List[int]) -> float:
        """
        Calculate average of target values for given indices.
        Used for regression task.

        Args:
            data: Full dataset
            indices: Indices to consider
        Returns:
            Mean of target values
        """
        if not indices:
            return 0.0
        return np.mean(data[indices, -1])

    def _calculate_impurity(self, data: np.ndarray, indices: List[int]) -> float:
        """
        Calculate impurity measure based on task type.
        For regression: Uses RSS (Residual Sum of Squares)
        For classification: Uses Gini impurity

        Args:
            data: Full dataset
            indices: Indices to consider
        Returns:
            Impurity value
        """
        if not indices:
            return 0.0

        if self.task == "regression":
            # Calculate RSS for regression
            values = data[indices, -1]
            return np.sum((values - np.mean(values)) ** 2)
        else:
            # Calculate Gini impurity for classification
            values, counts = np.unique(data[indices, -1], return_counts=True)
            probabilities = counts / len(indices)
            return 1 - np.sum(probabilities**2)

    def _generate(self, data: np.ndarray, indices: List[int], depth: int):
        """
        Generate the best split for the current node.

        Args:
            data: Full dataset
            indices: Indices to consider for splitting
            depth: Current depth in the tree

        Returns:
            Either False if no valid split found, or tuple containing:
            - Feature index to split on
            - Threshold value for split
            - Left subtree
            - Right subtree
        """
        # Get number of features (excluding target column)
        n_features = data.shape[1] - 1

        # Randomly select subset of features to consider (Random Forest modification)
        m = np.random.choice(n_features, int(np.ceil(n_features / 3)), replace=False)

        if self.task == "regression":
            # Initialize optimal RSS value for regression
            labels = data[indices, -1]
            opt = (np.max(labels) - np.min(labels)) ** 2 * len(indices) + 1.0
        else:
            # Initialize optimal Gini impurity for classification
            opt = float("inf")

        best_split = None
        # Try each selected feature for splitting
        for j in m:
            # Get unique values in this feature for potential splits
            feature_values = np.unique(data[indices, j])

            # Try each value as a splitting threshold
            for cut in feature_values:
                # Split data into left and right based on threshold
                left_indices = [i for i in indices if data[i, j] <= cut]
                right_indices = [i for i in indices if data[i, j] > cut]

                # Skip if either split is empty
                if not left_indices or not right_indices:
                    continue

                # Calculate total impurity for this split
                tmp = self._calculate_impurity(
                    data, left_indices
                ) + self._calculate_impurity(data, right_indices)

                # Update best split if this split is better
                if tmp < opt:
                    opt = tmp
                    best_split = (j, cut, left_indices, right_indices)

        # If no valid split found, return False
        if not best_split:
            return False

        # Create child nodes with the best split
        attr, split, left_indices, right_indices = best_split
        return (
            attr,
            split,
            DecisionTree(data, left_indices, depth - 1, self.task),
            DecisionTree(data, right_indices, depth - 1, self.task),
        )

    def predict(self, x: np.ndarray) -> Union[float, int]:
        """
        Predict target value for a single sample.

        Args:
            x: Single sample features
        Returns:
            Predicted value (float for regression, int for classification)
        """
        # If leaf node, return stored prediction
        if self.leaf:
            return self.prediction
        # Otherwise, traverse left or right based on split
        return self.L.predict(x) if x[self.attr] <= self.split else self.R.predict(x)
