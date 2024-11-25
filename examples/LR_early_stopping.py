import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.models.logistic_regression import LogisticRegression


def load_heart_data():
    columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target",
    ]

    data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        names=columns,
        na_values="?",
    )

    data = data.dropna()
    data["target"] = (data["target"] > 0).astype(int)

    X = data.drop("target", axis=1).values
    y = data["target"].values
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_idx, train_idx = indices[:test_size], indices[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# Load and prepare data
X, y = load_heart_data()

# Parameters that worked well
ni = 0.01  # Higher learning rate that worked well
t_epochs = 2000
lambda_range = [-0.1, 0.1]

all_results = []
n_experiments = 10

for exp in range(n_experiments):
    X_train, X_test, y_train, y_test = split_data(X, y, random_state=exp)

    trial_results = []
    for trial in range(10):
        lr = LogisticRegression(learning_rate=ni, n_iterations=t_epochs)
        lr.lambd = (
            np.random.uniform(
                low=lambda_range[0], high=lambda_range[1], size=(X_train.shape[1] + 1)
            )
            .astype(float)
            .reshape(-1, 1)
        )

        # Using your existing early stopping mechanism
        lr.train(X_train, y_train, calc_error=True, early_stop="average10")

        y_pred_train = lr.predict(X_train)
        y_pred_test = lr.predict(X_test)

        train_acc = np.mean(y_pred_train == y_train)
        test_acc = np.mean(y_pred_test == y_test)

        trial_results.append(
            {
                "train_acc": train_acc,
                "test_acc": test_acc,
                "n_iterations": len(lr.training_er),
                "final_loss": lr.loss[-1],
            }
        )

    all_results.extend(trial_results)

results_df = pd.DataFrame(all_results)

# Plotting
plt.figure(figsize=(15, 10))

# 1. Accuracy vs Iterations Scatter
plt.subplot(2, 2, 1)
plt.scatter(results_df["n_iterations"], results_df["test_acc"], alpha=0.6)
plt.xlabel("Number of Iterations")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy vs Number of Iterations")

# 2. Loss vs Accuracy
plt.subplot(2, 2, 2)
plt.scatter(results_df["final_loss"], results_df["test_acc"], alpha=0.6)
plt.xlabel("Final Loss")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy vs Final Loss")

# 3. Distribution of Accuracies
plt.subplot(2, 2, 3)
sns.histplot(data=results_df, x="test_acc", bins=15)
plt.xlabel("Test Accuracy")
plt.title("Distribution of Test Accuracies")

# 4. Distribution of Iterations
plt.subplot(2, 2, 4)
sns.histplot(data=results_df, x="n_iterations", bins=15)
plt.xlabel("Number of Iterations")
plt.title("Distribution of Training Iterations")

plt.tight_layout()
plt.show()

# Print detailed statistics
print("\nDetailed Statistics:")
print("\nAccuracy Distribution:")
print(results_df["test_acc"].describe())

print("\nIterations Distribution:")
print(results_df["n_iterations"].describe())

# Calculate quartiles for iterations and corresponding accuracies
iter_quartiles = np.percentile(results_df["n_iterations"], [25, 50, 75])
print("\nIteration Quartiles:")
print(f"25th percentile: {iter_quartiles[0]:.0f}")
print(f"Median: {iter_quartiles[1]:.0f}")
print(f"75th percentile: {iter_quartiles[2]:.0f}")


# Group results by iteration ranges
def get_iter_range(n):
    if n < iter_quartiles[0]:
        return "Very Short"
    elif n < iter_quartiles[1]:
        return "Short"
    elif n < iter_quartiles[2]:
        return "Medium"
    else:
        return "Long"


results_df["iter_range"] = results_df["n_iterations"].apply(get_iter_range)
print("\nAccuracy by Training Length:")
print(results_df.groupby("iter_range")["test_acc"].describe())

# Find optimal ranges
best_models = results_df[results_df["test_acc"] > results_df["test_acc"].quantile(0.75)]
print("\nCharacteristics of Top 25% Models:")
print(f"Average iterations: {best_models['n_iterations'].mean():.0f}")
print(f"Average final loss: {best_models['final_loss'].mean():.4f}")
print(f"Average test accuracy: {best_models['test_acc'].mean():.4f}")
