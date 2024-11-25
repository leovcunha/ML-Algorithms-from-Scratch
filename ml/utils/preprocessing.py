import numpy as np

# normalize X data by column


def normalize_features(X: np.ndarray) -> np.ndarray:
    """
    Normalize features using min-max scaling.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix to normalize

    Returns:
    --------
    np.ndarray
        Normalized feature matrix
    """
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
