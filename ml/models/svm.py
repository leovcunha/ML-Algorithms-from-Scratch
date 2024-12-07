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
        """
        Trains an SVM classifier using simplified SMO algorithm.
        """
        self.kernel_function = kernel_function
        m, n = X.shape

        # Initialize variables
        self.alphas = np.zeros(m)
        self.b = 0
        E = np.zeros(m)
        passes = 0

        # Map 0 to -1
        y = np.where(y == 0, -1, y)

        # Pre-compute Kernel Matrix
        if kernel_function.__name__ == "linear_kernel":
            K = np.dot(X, X.T)
        elif "gaussian_kernel" in kernel_function.__name__:
            X2 = np.sum(X**2, axis=1)
            K = X2.reshape(-1, 1) + X2 - 2 * np.dot(X, X.T)
            K = np.exp(-K)
        else:
            K = np.zeros((m, m))
            for i in range(m):
                for j in range(i, m):
                    K[i, j] = kernel_function(X[i, :], X[j, :])
                    K[j, i] = K[i, j]

        print("\nTraining ...")
        dots = 12
        while passes < self.max_passes:
            num_changed_alphas = 0
            for i in range(m):
                E[i] = self.b + np.sum(self.alphas * y * K[:, i]) - y[i]

                if (y[i] * E[i] < -self.tol and self.alphas[i] < self.C) or (
                    y[i] * E[i] > self.tol and self.alphas[i] > 0
                ):

                    j = i
                    while j == i:
                        j = np.random.randint(0, m)

                    E[j] = self.b + np.sum(self.alphas * y * K[:, j]) - y[j]

                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]

                    if y[i] == y[j]:
                        L = max(0, self.alphas[j] + self.alphas[i] - self.C)
                        H = min(self.C, self.alphas[j] + self.alphas[i])
                    else:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])

                    if L == H:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    self.alphas[j] = self.alphas[j] - (y[j] * (E[i] - E[j])) / eta
                    self.alphas[j] = min(H, self.alphas[j])
                    self.alphas[j] = max(L, self.alphas[j])

                    if abs(self.alphas[j] - alpha_j_old) < self.tol:
                        self.alphas[j] = alpha_j_old
                        continue

                    self.alphas[i] = self.alphas[i] + y[i] * y[j] * (
                        alpha_j_old - self.alphas[j]
                    )

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

                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

            print(".", end="", flush=True)
            dots += 1
            if dots > 78:
                dots = 0
                print()

        print(" Done!\n")

        # Save the model
        idx = self.alphas > 0
        self.X = X[idx]
        self.y = y[idx]
        self.alphas = self.alphas[idx]

        # Fix for the weight calculation
        if kernel_function.__name__ == "linear_kernel":
            self.w = np.zeros(n)
            for i in range(len(self.alphas)):
                self.w += self.alphas[i] * self.y[i] * self.X[i]

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
