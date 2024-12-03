import numpy as np


class SVM:
    def __init__(self, C=1.0, tol=1e-3, max_passes=5):
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.X = None
        self.y = None
        self.alphas = None
        self.b = 0
        self.w = None
        self.kernel_function = None

    def train(self, X, y, kernel_function):
        # Setup initial variables
        m, n = X.shape  # m = number of examples, n = number of features
        self.alphas = np.zeros(m)  # Lagrange multipliers, initially all zero
        self.b = 0  # bias term
        E = np.zeros(m)  # Error cache
        passes = 0  # Count of passes through the dataset

        # Convert labels from 0/1 to -1/+1 for SVM mathematics
        y = np.where(y == 0, -1, y)

        # Pre-compute kernel matrix for efficiency
        K = np.zeros((m, m))
        for i in range(m):
            for j in range(i, m):
                K[i, j] = kernel_function(X[i, :], X[j, :])
                K[j, i] = K[i, j]  # Kernel matrix is symmetric

        # Main training loop
        while passes < self.max_passes:
            num_changed_alphas = 0  # Track if we made any updates in this pass

            # Loop through all training examples
            for i in range(m):
                # Calculate error for example i
                # E[i] = f(x_i) - y_i where f(x_i) is the current SVM output
                E[i] = self.b + np.sum(self.alphas * y * K[:, i]) - y[i]

                # Check if example i violates KKT conditions
                # This is where we select examples that need updating
                if (y[i] * E[i] < -self.tol and self.alphas[i] < self.C) or (
                    y[i] * E[i] > self.tol and self.alphas[i] > 0
                ):

                    # Select second example j randomly
                    j = np.random.randint(0, m)
                    while j == i:
                        j = np.random.randint(0, m)

                    # Calculate error for example j
                    E[j] = self.b + np.sum(self.alphas * y * K[:, j]) - y[j]

                    # Save old alpha values
                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]

                    # Calculate bounds L and H
                    # These ensure alphas stay between 0 and C
                    if y[i] == y[j]:
                        L = max(0, self.alphas[j] + self.alphas[i] - self.C)
                        H = min(self.C, self.alphas[j] + self.alphas[i])
                    else:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])

                    if L == H:
                        continue

                    # Calculate eta (the second derivative of the objective)
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    # Update alpha_j
                    self.alphas[j] -= (y[j] * (E[i] - E[j])) / eta

                    # Clip alpha_j to be between L and H
                    self.alphas[j] = min(H, self.alphas[j])
                    self.alphas[j] = max(L, self.alphas[j])

                    # If alpha_j didn't move enough, continue to next example
                    if abs(self.alphas[j] - alpha_j_old) < self.tol:
                        self.alphas[j] = alpha_j_old
                        continue

                    # Update alpha_i based on alpha_j's update
                    self.alphas[i] += y[i] * y[j] * (alpha_j_old - self.alphas[j])

                    # Update threshold b
                    b1 = (
                        self.b
                        - E[i]
                        - y[i] * (self.alphas[i] - alpha_i_old) * K[i, j]
                        - y[j] * (self.alphas[j] - alpha_j_old) * K[i, j]
                    )

                    b2 = (
                        self.b
                        - E[j]
                        - y[i] * (self.alphas[i] - alpha_i_old) * K[i, j]
                        - y[j] * (self.alphas[j] - alpha_j_old) * K[j, j]
                    )

                    # Set new threshold
                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    num_changed_alphas += 1

            # Update passes count
            if num_changed_alphas == 0:
                passes += 1  # No updates made, increment passes
            else:
                passes = 0  # Updates made, reset passes count

    def predict(self, X):
        if self.kernel_function.__name__ == "linear_kernel":
            return np.sign(np.dot(X, self.w) + self.b)
        else:
            m = X.shape[0]
            pred = np.zeros(m)
            for i in range(m):
                prediction = 0
                for j in range(len(self.alphas)):
                    prediction += (
                        self.alphas[j]
                        * self.y[j]
                        * self.kernel_function(X[i, :], self.X[j, :])
                    )
                pred[i] = prediction
            return np.sign(pred + self.b)


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def gaussian_kernel(x1, x2, sigma=2.0):
    return np.exp(-np.sum((x1 - x2) ** 2) / (2 * (sigma**2)))
