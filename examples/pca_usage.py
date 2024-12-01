from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path to import from ml.models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.models.pca import PCA


def plot_digits(original, reconstructed, n_samples=5):
    """Helper function to plot original and reconstructed digits"""
    plt.figure(figsize=(10, 4))
    for i in range(n_samples):
        # Original digit
        plt.subplot(2, n_samples, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        if i == 0:
            plt.title("Original Digits")

        # Reconstructed digit
        plt.subplot(2, n_samples, n_samples + i + 1)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        if i == 0:
            plt.title("Reconstructed Digits")
    plt.tight_layout()


def main():
    # Load MNIST
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X[:1000]  # Use first 1000 samples for demonstration

    # Scale data to [0,1]
    X = X / 255.0

    # Initialize PCA
    pca = PCA()

    # Center the data
    X_centered, mu = pca.subtract_mean(X)

    # Fit PCA
    U, S = pca.fit(X_centered)

    # Project onto reduced dimension
    k = 50  # Number of components to keep
    Z = pca.project_data(X_centered, U, k)

    # Recover the data
    X_recovered = pca.recover_data(Z, U, k, mu)

    # Calculate explained variance
    # to determine how many components we need to retain to preserve a certain percentage of the data's variance
    total_variance = np.sum(S)
    explained_variance_ratio = S[:k] / total_variance
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    print(f"Original data shape: {X.shape}")
    print(f"Projected data shape: {Z.shape}")
    print(f"Variance explained by {k} components: {cumulative_variance_ratio[-1]:.2%}")

    # Plotting
    plot_digits(X, X_recovered)

    # Plot explained variance
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(explained_variance_ratio[:50], "bo-")
    plt.title("Explained Variance Ratio")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")

    plt.subplot(1, 2, 2)
    plt.plot(cumulative_variance_ratio[:50], "ro-")
    plt.title("Cumulative Explained Variance Ratio")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance Ratio")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
