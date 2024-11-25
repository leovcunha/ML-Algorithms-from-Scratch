import numpy as np


from ml.utils.activations import sigmoid
from typing import Optional, Tuple


class LogisticRegression:
    """
    Logistic Regression classifier implemented from scratch with gradient descent.

    Features:
    - Gradient descent optimization
    - Lambda regularization
    - Early stopping options
    - Error calculation during training

    Parameters:
    -----------
    learning_rate : float, default=0.001
        Step size for gradient descent
    n_iterations : int, default=1000
        Maximum number of training iterations
    """

    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambd = []
        self.loss = []
        self.training_er = []  # Training error history

        # Initialize internal variables
        self.X = None
        self.y = None
        self.classes = None
        self.training_size = None
        self.n_features = None
        self.is_fitted = False

    def train(self, X, y, calc_error=False, early_stop=None, lambda_range=None):
        """
        Train the logistic regression model.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        calc_error : bool, default=False
            Whether to calculate training error at each iteration
        early_stop : str, optional
            Early stopping strategy ('average10' or 'validation')
        lambda_range : tuple of float, optional
            Range for random initialization of lambda parameters
        """
        self.X = np.array(X)
        self.y = np.array(y)
        self.classes = np.unique(y)
        # size of the training set
        self.training_size = self.X.shape[0]
        # number of features accounting +1 for the constant term lambd_0 , as lambd will have D+1
        self.n_features = self.X.shape[1] + 1
        # to allow proper multiplication with the term added above, added a vector of ones in first column of X
        self.X = np.concatenate((np.ones([self.training_size, 1]), self.X), axis=1)
        # initialize vector of lambdas with random values if not predefined values or range
        self._initialize_lambda(lambda_range)

        # training loop with gradient descent
        # Training loop
        self._train_loop(calc_error, early_stop)

        self.is_fitted = True

    def _train_loop(self, calc_error: bool, early_stop: Optional[str]) -> None:
        """Main training loop with gradient descent and early stopping."""
        self._compute_loss()

        for i in range(self.n_iterations):
            self.lambd = self.lambd - self.learning_rate * self._compute_gradient()
            self._compute_loss()

            if calc_error:
                predictions = self._compute_predictions()
                predictions = (predictions.T > 0.5).astype(int)
                self.training_er.append(self.compute_error(self.y, predictions))

                if early_stop == "average10" and len(self.training_er) >= 20:
                    if self._should_stop_early():
                        print(f"Early stopping at iteration {i}")
                        break

    def _compute_loss(self) -> None:
        """Calculate binary cross-entropy loss."""
        predictions = self._compute_predictions()
        epsilon = 1e-15  # Small constant to prevent log(0)
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -np.mean(
            self.y * np.log(predictions.T) + (1 - self.y) * np.log(1 - predictions.T)
        )

        self.loss.append(loss)

    def _initialize_lambda(self, lambda_range: Optional[Tuple[float, float]]) -> None:
        """Initialize model parameters (lambda) either randomly or within specified range."""
        if not np.any(self.lambd):
            if lambda_range:
                self.lambd = (
                    np.random.uniform(
                        low=lambda_range[0],
                        high=lambda_range[1],
                        size=(self.n_features),
                    )
                    .astype(float)
                    .reshape(self.n_features, -1)
                )
            else:
                self.lambd = np.random.rand(self.n_features, 1)

    def _compute_predictions(self):
        return sigmoid(np.dot(self.X, self.lambd))

    def _should_stop_early(self) -> bool:
        """Check if early stopping criteria is met based on training error."""
        if len(self.training_er) < 20:
            return False

        # Original logic: stop if average of last 10 errors is not improving by at least 1%
        return np.mean(self.training_er[-10:]) > 0.99 * np.mean(
            self.training_er[-20:-10]
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in features.

        Parameters:
        -----------
        features : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")

        X = np.concatenate((np.ones([features.shape[0], 1]), features), axis=1)
        probabilities = sigmoid(np.dot(X, self.lambd))
        predictions = (probabilities > 0.5).astype(int)
        return predictions.T

    def compute_error(self, y, ypred):
        """
        compute error rate
        """
        return np.sum((y - ypred) ** 2)

    def _compute_gradient(self) -> np.ndarray:
        """Calculate gradient for gradient descent."""
        error = self._compute_predictions().T - self.y
        gradient = np.dot(error, self.X) / len(self.y)
        return gradient.reshape(len(self.lambd), -1)
