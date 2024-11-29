# examples/hierarchical_clustering_usage.py

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.models.hierarchical_clustering import (
    AgglomerativeClustering,
    single_linkage,
    complete_linkage,
    average_linkage,
    centroid_linkage,
)


def plot_dendrogram_from_scratch(hc, title):
    """Plot dendrogram using our own hierarchical clustering implementation"""
    plt.figure(figsize=(10, 7))

    def get_plot_coordinates(node, leaf_locations):
        """Recursively get coordinates for plotting"""
        if not node.left and not node.right:  # Leaf node
            x = leaf_locations[node.get_obs()[0]]
            return x, 0, x

        # Get coordinates of children
        left_x, left_y, left_center = get_plot_coordinates(node.left, leaf_locations)
        right_x, right_y, right_center = get_plot_coordinates(
            node.right, leaf_locations
        )

        # Plot vertical lines from children to merge height
        plt.vlines(x=left_center, ymin=left_y, ymax=node.height, color="blue")
        plt.vlines(x=right_center, ymin=right_y, ymax=node.height, color="blue")

        # Plot horizontal line connecting the clusters
        plt.hlines(y=node.height, xmin=left_center, xmax=right_center, color="blue")

        return min(left_x, right_x), node.height, (left_center + right_center) / 2

    # Create evenly spaced x-coordinates for leaf nodes
    n_leaves = len(hc.data)
    leaf_locations = {i: i for i in range(n_leaves)}

    # Plot the dendrogram structure
    root = hc.clusters[hc.levels][0]
    get_plot_coordinates(root, leaf_locations)

    # Plot leaf nodes
    for i in range(n_leaves):
        plt.plot(i, 0, "ro")  # Red dots for leaf nodes
        plt.text(i, -0.5, str(i), ha="center", va="top")

    plt.title(title)
    plt.xlabel("Data Points")
    plt.ylabel("Merge Distance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    # Create example dataset
    np.random.seed(42)

    # dataset for dendrogram visualization
    X = np.array(
        [
            [0, 0],  # Point 0
            [0.5, 0],  # Point 1
            [0, 0.5],  # Point 2
            [4, 4],  # Point 3
            [4.2, 4],  # Point 4
            [4, 4.2],  # Point 5
            [8, 8],  # Point 6
        ]
    )

    # Try different linkage methods
    linkage_methods = {
        "Single Linkage": single_linkage,
        "Complete Linkage": complete_linkage,
        "Average Linkage": average_linkage,
        "Centroid Linkage": centroid_linkage,
    }

    for name, linkage in linkage_methods.items():
        # Create and fit clustering model using my implementation
        hc = AgglomerativeClustering(X, linkage=linkage)

        # Plot dendrogram using my implementation
        plot_dendrogram_from_scratch(hc, f"Hierarchical Clustering Dendrogram - {name}")

        # Print merge information
        print(f"\n{name} Merge Sequence:")
        for level in range(hc.levels + 1):
            clusters = hc.clusters[level]
            print(f"Level {level}: {[node.get_obs() for node in clusters]}")


if __name__ == "__main__":
    main()
