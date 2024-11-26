import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from ..ml.models.random_forest import RandomForest

# Set random seed for reproducibility
RANDOM_STATE = 2404
np.random.seed(RANDOM_STATE)

# Load Boston housing dataset
boston = load_boston()
X, y = boston.data, boston.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# Define hyperparameters
n_trees = 100  # Number of trees in forest
n_samples = len(X_train)  # Number of training samples
height = [3, 5, 7]  # Different tree heights to try
B_range = [10, 50, 100]  # Different numbers of trees to try

# Dictionaries to store results
train_mse_dict = {}
test_mse_dict = {}

# Try different combinations of height and number of trees
for h in height:
    train_mse_dict[h] = []
    test_mse_dict[h] = []
    for B in B_range:
        # Generate bootstrap samples for each tree
        bootstrap_indices = [
            np.random.choice(n_samples, n_samples, replace=True) for _ in range(B)
        ]

        # Create and train random forest
        rforest = RandomForest(X_train, y_train, bootstrap_indices, h)

        # Calculate training and testing MSE
        tr_mse = rforest.get_mse(X_train, y_train)
        ts_mse = rforest.get_mse(X_test, y_test)

        # Print results
        print(f"height: {h}, B: {B}\n train MSE: {tr_mse:.4f}\n test MSE: {ts_mse:.4f}")

        # Store results
        train_mse_dict[h].append(tr_mse)
        test_mse_dict[h].append(ts_mse)
