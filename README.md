# Machine Learning Algorithms from Scratch

A Python implementation of machine learning algorithms built from scratch using NumPy. This project focuses on transparency and educational value by implementing algorithms without using high-level ML libraries.

## 🎯 Models

-   **Logistic Regression Implementation**

    -   Gradient Descent optimization
    -   Early stopping mechanism
    -   Lambda regularization
    -   Configurable learning rates
    -   Grid search for hyperparameter tuning

    **Random Forest**

    -   Supports both classification and regression tasks
    -   Decision tree-based ensemble learning
    -   Bootstrap aggregating (bagging)
    -   Configurable tree height
    -   Error calculation (MSE for regression, Error rate for classification)

## 🛠️ Installation

```
# Clone the repository
git clone https://github.com/yourusername/ml-algorithms-from-scratch.git
# Install dependencies
pip install -r requirements.txt
```

### Run all tests

`pytest`

## Examples

This directory contains example scripts demonstrating how to use the implemented algorithms.

### Random Forest Example

The `random_forest_usage.py` script demonstrates both classification and regression tasks using the Random Forest implementation.

```
python examples/random_forest_usage.py
```

This will run:

1. Classification example using the Iris dataset
1. Regression example using the California Housing dataset (first 1000 samples)

### Logistic Regression Example

The `logistic_regression_usage.py` script demonstrates binary classification using the Cleveland Heart Disease dataset from UCI Machine Learning Repository.

```
python examples/logistic_regression_usage.py
```

Dataset details:

-   Source: UCI ML Repository (Cleveland Heart Disease)
-   Features: 13 medical attributes (age, sex, chest pain type, blood pressure, etc.)
-   Target: Binary classification (presence of heart disease)
-   Size: 303 instances (after cleaning)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
