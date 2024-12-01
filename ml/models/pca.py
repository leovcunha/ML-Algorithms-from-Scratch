import numpy as np


class PCA:
    """
    Principal Component Analysis implementation.

    Methods:
        fit: Computes principal components and eigenvalues
        subtract_mean: Centers the data by subtracting the mean
        project_data: Projects data onto k principal components
        recover_data: Recovers data from projection
    """

    def __init__(self):
        self.U = None
        self.S = None
        self.mu = None

    def fit(self, matrix):
        """
        Perform Principal Component Analysis (PCA)

        Args:
            matrix: Input data matrix where rows are samples and columns are features

        Returns:
            U: Principal components (eigenvectors of the covariance matrix)
            S: Eigenvalues in descending order
        """
        # Calculate the covariance matrix
        co_matrix = np.cov(matrix.T)

        # Calculate eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(co_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        idx = eigenvalues.argsort()[::-1]
        S = eigenvalues[idx]
        U = eigenvectors[:, idx]

        # Ensure the output is real
        U = np.real(U)
        S = np.real(S)

        self.U = U
        self.S = S

        return U, S

    def subtract_mean(self, X):
        """
        Subtract mean from data

        Args:
            X: Input data matrix

        Returns:
            X_mu: Mean-centered data
            mu: Mean that was subtracted
        """
        mu = np.mean(X, axis=0)
        X_mu = X - mu
        self.mu = mu
        return X_mu, mu

    def project_data(self, X_mu, U, k):
        """
        Project data onto k principal components

        Args:
            X_mu: Mean-centered data
            U: Principal components
            k: Number of components to project onto

        Returns:
            Z: Projected data
        """
        U_reduce = U[:, :k]
        Z = np.dot(X_mu, U_reduce)
        return Z

    def recover_data(self, Z, U, k, mu):
        """
        Recover data from projection

        Args:
            Z: Projected data
            U: Principal components
            k: Number of components used
            mu: Mean that was subtracted

        Returns:
            X_rec: Recovered data
        """
        U_reduce = U[:, :k]
        X_rec = np.dot(Z, U_reduce.T) + mu
        return X_rec
