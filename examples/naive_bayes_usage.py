import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Add the parent directory to the path to import from ml.models
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.models.naive_bayes import NaiveBayes


def main():
    # Generate a synthetic dataset
    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        random_state=42,
        shuffle=False,
    )

    # Split the data into training and test sets
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    # Create and train the Naive Bayes model
    nb = NaiveBayes()
    nb.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = nb.predict(X_test)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy:.2f}")

    # Plot decision boundary
    h = 0.02  # step size in mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = nb.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.viridis)

    plt.scatter(
        X_train[:, 0], X_train[:, 1], c=y_train, edgecolor="k", s=20, cmap="viridis"
    )
    plt.title("Naive Bayes Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


if __name__ == "__main__":
    main()
