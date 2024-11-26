import numpy as np
from typing import List, Optional, Tuple, Union


class DecisionTree:
    """
    Decision Tree implementation using RSS (Residual Sum of Squares) for splitting.
    This implementation is specifically designed for regression problems.
    """

    def __init__(self, data: np.ndarray, indices: List[int], depth: int):
        """
        Initialize and build the decision tree recursively.

        Args:
            data: Combined features and target array where last column is the target
            indices: List of indices to use from data (supports bootstrap sampling)
            depth: Maximum allowed depth of the tree
        """
        # Initialize node attributes
        self.leaf = False  # Flag to indicate if node is a leaf
        self.prediction = None  # Store prediction value for leaf nodes
        self.attr = None  # Store splitting attribute/feature index
        self.split = None  # Store splitting value/threshold
        self.L = None  # Left subtree
        self.R = None  # Right subtree

        # Case 1: If we've reached maximum depth, make this a leaf node
        if depth == 0:
            self.leaf = True
            self.prediction = self._avg(data, indices)

        # Case 2: If all samples have same target value, make this a leaf node
        elif len(set(data[indices, -1])) == 1:
            self.leaf = True
            self.prediction = data[indices[0], -1]

        # Case 3: Try to split the node
        else:
            tree_info = self._generate(data, indices, depth)
            # If splitting fails, make this a leaf node
            if not tree_info:
                self.leaf = True
                self.prediction = self._avg(data, indices)
            # If splitting succeeds, create internal node with left and right children
            else:
                self.attr, self.split, self.L, self.R = tree_info

    @staticmethod
    def _avg(data: np.ndarray, indices: List[int]) -> float:
        """
        Calculate average of target values for given indices.

        Args:
            data: Full dataset
            indices: Indices to consider
        Returns:
            Mean of target values
        """
        if not indices:
            return 0.0
        return np.mean(data[indices, -1])

    @staticmethod
    def _rss(data: np.ndarray, indices: List[int]) -> float:
        """
        Calculate Residual Sum of Squares (RSS) for given indices.
        RSS = Σ(y - ȳ)² where ȳ is the mean of y values

        Args:
            data: Full dataset
            indices: Indices to consider
        Returns:
            RSS value
        """
        if not indices:
            return 0.0
        values = data[indices, -1]
        return np.sum((values - np.mean(values)) ** 2)

    def _generate(
        self, data: np.ndarray, indices: List[int], depth: int
    ) -> Union[Tuple[int, float, "DecisionTree", "DecisionTree"], bool]:
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

        # Initialize optimal RSS value
        labels = data[indices, -1]
        opt = (np.max(labels) - np.min(labels)) ** 2 * len(indices) + 1.0

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

                # Calculate total RSS for this split
                tmp = self._rss(data, left_indices) + self._rss(data, right_indices)

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
            DecisionTree(data, left_indices, depth - 1),
            DecisionTree(data, right_indices, depth - 1),
        )

    def predict(self, x: np.ndarray) -> float:
        """
        Predict target value for a single sample.

        Args:
            x: Single sample features
        Returns:
            Predicted value
        """
        # If leaf node, return stored prediction
        if self.leaf:
            return self.prediction
        # Otherwise, traverse left or right based on split
        return self.L.predict(x) if x[self.attr] <= self.split else self.R.predict(x)
