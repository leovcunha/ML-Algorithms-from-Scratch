import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification  # Ensure this is correctly imported
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.models.multilayer_perceptron import MultilayerPerceptron


def plot_decision_boundary(model, X, y, n_classes):
    # Set min and max values with some padding
    x_min, x_max = X[0].min() - 0.5, X[0].max() + 0.5
    y_min, y_max = X[1].min() - 0.5, X[1].max() + 0.5

    # Create a finer mesh grid for smoother boundary
    h = 0.005  # Decreased step size for better resolution
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Flatten the grid and make predictions
    grid_points = np.c_[xx.ravel(), yy.ravel()].T  # Shape: (2, n_points)
    Z = model.predict(grid_points)  # Predict on all grid points
    Z = np.argmax(Z, axis=0)  # Convert probabilities to class labels

    # Reshape Z to match the shape of the grid
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.rainbow)
    scatter = plt.scatter(
        X[0], X[1], c=np.argmax(y, axis=0), cmap=plt.cm.rainbow, edgecolors="black"
    )
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary")
    plt.colorbar(scatter)
    plt.show()


def main():
    # Generate a complex 3-class dataset
    n_classes = 3
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_informative=2,  # All features are informative
        n_redundant=0,  # No redundant features
        n_repeated=0,  # No repeated features
        n_classes=n_classes,
        n_clusters_per_class=1,  # Reduce the number of clusters per class to fit constraints
        random_state=42,
    )
    X = X.T
    y = y.reshape(1, -1)

    # One-hot encode labels
    Y = np.eye(n_classes)[y.reshape(-1)].T  # Shape: (n_classes, n_samples)

    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X.T, Y.T, test_size=0.2, random_state=42
    )
    X_train, X_test = X_train.T, X_test.T
    Y_train, Y_test = Y_train.T, Y_test.T

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.T).T
    X_test = scaler.transform(X_test.T).T

    # Create and train the model
    mlp = MultilayerPerceptron(
        layer_dims=[
            2,
            20,
            20,
            20,
            20,
            n_classes,
        ],  # Output layer matches the number of classes
        learning_rate=0.01,
        batch_size=64,
        n_iterations=10000,
        random_seed=42,
        verbose=True,
    )

    # Train the model
    costs = mlp.fit(X_train, Y_train)

    # Plot training costs
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(costs)) * 100, costs)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Training Cost over Iterations")
    plt.grid(True)
    plt.show()

    # Plot decision boundary
    plot_decision_boundary(mlp, X_test, Y_test, n_classes)

    # Calculate and print accuracy
    predictions = mlp.predict(X_test)
    predictions = np.argmax(
        predictions, axis=0
    )  # Convert probabilities to class labels
    actual = np.argmax(Y_test, axis=0)  # Convert one-hot to class labels
    accuracy = np.mean(predictions == actual)
    print(f"Test accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
