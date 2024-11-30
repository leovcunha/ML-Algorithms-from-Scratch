import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path to import from ml.models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.models.KNN import KNN


def plot_2d_classifier(X, y, classifier, title):
    """
    Utility function to visualize the decision boundary of a 2D classifier
    """
    # Create a mesh grid
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict for each point in mesh
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the result
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title(title)
    plt.show()


def main():
    # Example 1: Binary Classification
    print("\n=== Binary Classification Example ===")

    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=100, n_features=2, n_classes=2, n_redundant=0, random_state=42
    )

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train and evaluate KNN with different k values
    k_values = [1, 3, 5]
    for k in k_values:
        knn = KNN(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        accuracy = knn.score(X_test_scaled, y_test)
        print(f"Accuracy with k={k}: {accuracy:.2f}")

        # Plot decision boundary
        plot_2d_classifier(
            X_train_scaled, y_train, knn, f"KNN Decision Boundary (k={k})"
        )

    # Example 2: Iris Dataset (Multiclass)
    print("\n=== Iris Dataset Example ===")

    # Load and prepare iris dataset
    iris = load_iris()
    X = iris.data[:, [0, 2]]  # using only two features for visualization
    y = iris.target

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train and evaluate
    knn = KNN(n_neighbors=3)
    knn.fit(X_train_scaled, y_train)
    accuracy = knn.score(X_test_scaled, y_test)
    print(f"Iris dataset accuracy: {accuracy:.2f}")

    # Plot results
    plot_2d_classifier(X_train_scaled, y_train, knn, "KNN on Iris Dataset (k=3)")

    # Example 3: Single Prediction
    print("\n=== Single Prediction Example ===")

    # Make a prediction for a single sample
    sample = X_test_scaled[0].reshape(1, -1)
    prediction = knn.predict(sample)
    true_label = y_test[0]
    print(f"Predicted class: {prediction[0]}")
    print(f"True class: {true_label}")


if __name__ == "__main__":
    main()
