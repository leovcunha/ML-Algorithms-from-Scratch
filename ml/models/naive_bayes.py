import numpy as np


class NaiveBayes:
    """
    A simple Naive Bayes classifier.
    """

    def __init__(self):
        self.classes = None
        self.priors = None
        self.means = None
        self.variances = None

    def fit(self, X, y):
        """
        Fits the Naive Bayes model to the training data.

        Args:
            X (np.ndarray): Training data features.
            y (np.ndarray): Training data labels.
        """
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.priors = np.zeros(n_classes)
        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.priors[i] = X_c.shape[0] / X.shape[0]
            self.means[i, :] = np.mean(X_c, axis=0)
            self.variances[i, :] = np.var(X_c, axis=0)

    def predict(self, X):
        """
        Predicts the class labels for the given data.

        Args:
            X (np.ndarray): Data features to predict.

        Returns:
            np.ndarray: Predicted class labels.
        """
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        """
        Predicts the class label for a single data point.

        Args:
            x (np.ndarray): Single data point features.

        Returns:
            int: Predicted class label.
        """
        posteriors = []
        for i, c in enumerate(self.classes):
            prior = np.log(self.priors[i])
            class_conditional = np.sum(
                np.log(self._pdf(x, self.means[i, :], self.variances[i, :]))
            )
            posterior = prior + class_conditional
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def _pdf(self, x, mean, variance):
        """
        Calculates the probability density function for a Gaussian distribution.

        Args:
            x (np.ndarray): Data point features.
            mean (np.ndarray): Mean of the Gaussian distribution.
            variance (np.ndarray): Variance of the Gaussian distribution.

        Returns:
            np.ndarray: Probability density function values.
        """
        numerator = np.exp(-((x - mean) ** 2) / (2 * variance))
        denominator = np.sqrt(2 * np.pi * variance)
        return numerator / denominator
