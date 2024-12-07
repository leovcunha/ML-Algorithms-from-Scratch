import matplotlib.pyplot as plt
from sklearn import datasets
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.models.k_means_clustering import k_means_clustering

# Load a simple dataset
iris = datasets.load_iris()
X = iris.data

# Apply k-means clustering
n_clusters = 3
labels, centroids = k_means_clustering(X, k=n_clusters)

# Visualize the results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=50)
plt.title("K-Means Clustering on Digits Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
