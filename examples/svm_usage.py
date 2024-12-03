# examples/svm_usage.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import urllib.request
import os
from typing import Tuple
import matplotlib.pyplot as plt
from time import time

# Import the SVM from parent directory
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.models.svm import SVM, linear_kernel, gaussian_kernel


def fetch_spambase_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Download and prepare the Spambase dataset from UCI ML Repository.
    Returns preprocessed features and labels.
    """
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Download the dataset if it doesn't exist
    data_path = os.path.join(data_dir, "spambase.data")
    if not os.path.exists(data_path):
        print("Downloading Spambase dataset...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
        try:
            urllib.request.urlretrieve(url, data_path)
            print("Download completed!")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            # Provide a direct link if download fails
            print("Please download manually from:")
            print(url)
            print(f"And place it in: {data_path}")
            return None, None

    try:
        # Load and prepare the dataset
        data = np.loadtxt(data_path, delimiter=",")
        X = data[:, :-1]  # All columns except the last one
        y = data[:, -1]  # Last column is the label

        print(f"Dataset loaded: {X.shape[0]} samples with {X.shape[1]} features")
        return X, y

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None


def plot_training_results(
    X_test: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray, title: str
):
    """
    Plot confusion matrix and performance metrics.
    """
    from sklearn.metrics import confusion_matrix, classification_report

    # Create figure with subplots
    plt.style.use("default")
    fig = plt.figure(figsize=(15, 10))

    # 1. Confusion Matrix (top left)
    ax1 = plt.subplot(2, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    im = ax1.imshow(cm, interpolation="nearest", cmap="Blues")
    ax1.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax1)

    # Add labels
    classes = ["Not Spam (0)", "Spam (1)"]
    tick_marks = np.arange(len(classes))
    ax1.set_xticks(tick_marks)
    ax1.set_yticks(tick_marks)
    ax1.set_xticklabels(classes)
    ax1.set_yticklabels(classes)

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(
                j,
                i,
                f"{cm[i, j]}\n({cm[i, j]/np.sum(cm)*100:.1f}%)",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    # 2. Prediction Distribution (top right)
    ax2 = plt.subplot(2, 2, 2)
    ax2.hist([y_test, y_pred], label=["Actual", "Predicted"], bins=2, alpha=0.7)
    ax2.set_title("Distribution of Predictions vs Actual")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Not Spam", "Spam"])
    ax2.legend()

    # 3. Performance Metrics (bottom)
    ax3 = plt.subplot(2, 1, 2)
    report = classification_report(
        y_test, y_pred, target_names=["0", "1"], output_dict=True, zero_division=0
    )

    # Create table of metrics
    cell_text = []
    metrics_classes = ["0", "1"]  # Using numerical labels
    for label in metrics_classes:
        row = [
            f"{report[label]['precision']*100:.1f}%",
            f"{report[label]['recall']*100:.1f}%",
            f"{report[label]['f1-score']*100:.1f}%",
        ]
        cell_text.append(row)

    table = ax3.table(
        cellText=cell_text,
        rowLabels=["Not Spam (0)", "Spam (1)"],
        colLabels=["Precision", "Recall", "F1-Score"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    ax3.set_title("Performance Metrics")
    ax3.axis("off")

    # Overall title
    plt.suptitle(
        f"{title}\nAccuracy: {np.mean(y_pred == y_test):.2%}", fontsize=16, y=1.02
    )
    plt.tight_layout()
    plt.show()
    plt.close()


def main():
    # 1. Load and prepare data
    print("Loading Spambase dataset...")
    X, y = fetch_spambase_data()

    if X is None or y is None:
        print("Failed to load dataset. Exiting.")
        return

    # 2. Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use smaller subset for demonstration
    train_size = 1000
    print(f"\nUsing {train_size} samples for training...")

    # 4. Train and evaluate with linear kernel
    print("\nTraining SVM with linear kernel...")
    svm_linear = SVM(C=1.0, tol=1e-3, max_passes=5)
    start_time = time()

    # Convert labels to -1 and 1 for SVM
    y_train_svm = np.where(y_train[:train_size] == 0, -1, 1)
    y_test_svm = np.where(y_test == 0, -1, 1)

    svm_linear.train(
        X_train_scaled[:train_size],
        y_train_svm,
        kernel_function=linear_kernel,
    )

    training_time = time() - start_time
    y_pred_linear = svm_linear.predict(X_test_scaled)
    # Convert predictions back to 0 and 1
    y_pred_linear = np.where(y_pred_linear == -1, 0, 1)
    accuracy_linear = np.mean(y_pred_linear == y_test)

    print(f"\nLinear Kernel Results:")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Accuracy: {accuracy_linear:.2%}")

    # Plot linear kernel results
    plot_training_results(
        X_test_scaled, y_test, y_pred_linear, "Linear Kernel Classification Results"
    )


if __name__ == "__main__":
    main()
