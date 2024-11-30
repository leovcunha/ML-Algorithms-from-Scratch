import numpy as np
from collections import Counter


class KNN:
    """
    K-Nearest Neighbors classifier implementation
    Supports any value of k (number of neighbors)
    """

    def __init__(self, n_neighbors=3):
        """
        Initialize the KNN classifier

        Parameters:
        -----------
        n_neighbors : int, default=3
            Number of neighbors to use for classification
        """
        self.n_neighbors = n_neighbors
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")

    def fit(self, X, y):
        """
        Store the training data for later use in predictions
        KNN is a lazy learner - no actual training is performed

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data features
        y : array-like of shape (n_samples,)
            Target values (class labels)
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.classes = np.unique(y)

    def predict(self, X):
        """
        Predict class labels for samples in X

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns:
        --------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        # Calculate distances between each test point and all training points
        distances = [self.euclidean_distance(self.X_train, x) for x in X]

        # Get indices of k nearest neighbors for each test point
        k_nearest_indices = [self.get_k_nearest_indices(dist) for dist in distances]

        # Predict class by majority vote among k nearest neighbors
        return np.array([self.majority_vote(idx) for idx in k_nearest_indices])

    def euclidean_distance(self, x1, x2):
        """
        Calculate Euclidean distance between points

        Parameters:
        -----------
        x1 : array-like of shape (n_samples, n_features)
            First set of samples
        x2 : array-like of shape (n_features,)
            Single sample to compare against x1

        Returns:
        --------
        distances : array of shape (n_samples,)
            Euclidean distances between x2 and each sample in x1
        """
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

    def get_k_nearest_indices(self, distances):
        """
        Find indices of the k minimum distances

        Parameters:
        -----------
        distances : array-like
            Array of distances

        Returns:
        --------
        k_nearest_indices : array
            Indices of k nearest neighbors
        """
        # Use argsort to get indices of k smallest distances
        return np.argsort(distances)[: self.n_neighbors]

    def majority_vote(self, indices):
        """
        Determine class by majority vote among k nearest neighbors

        Parameters:
        -----------
        indices : array-like
            Indices of k nearest neighbors

        Returns:
        --------
        predicted_class :
            Most common class among neighbors
        """
        # Get labels of k nearest neighbors
        k_nearest_labels = self.y_train[indices]

        # Find most common class using Counter
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def score(self, X_test, y_test):
        """
        Calculate prediction accuracy on test set

        Parameters:
        -----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples
        y_test : array-like of shape (n_samples,)
            True labels for test samples

        Returns:
        --------
        accuracy : float
            Proportion of correct predictions
        """
        y_pred = self.predict(X_test)
        return np.mean(y_pred == y_test)
