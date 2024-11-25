import numpy as np
from ml.utils.preprocessing import preprocess_auto

X, y = preprocess_auto()


def split_data(X, y):
    seed = 244
    np.random.seed(seed)

    length = X.shape[0]

    nums = np.arange(X.shape[0])
    np.random.shuffle(nums)
    X_train = X[nums[0 : length // 2]]
    X_test = X[nums[length // 2 : length]]
    y_train = y[nums[0 : length // 2]]
    y_test = y[nums[length // 2 : length]]

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X, y = preprocess_auto()
    X_train, X_test, y_train, y_test = split_data(X, y)
    print("X train \n========\n", X_train)
    print("X test \n========\n", X_test)
    print("y train \n========\n", y_train)
    print("y test \n========\n", y_test)
