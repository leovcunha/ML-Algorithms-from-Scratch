# tests/test_hierarchical_clustering.py

import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.models.hierarchical_clustering import (
    AgglomerativeClustering,
    single_linkage,
    complete_linkage,
    average_linkage,
    centroid_linkage,
    Node,
)


@pytest.fixture
def sample_data():
    """Fixture for test data"""
    return np.array(
        [
            [0, 0],  # Point 0
            [0, 1],  # Point 1
            [4, 4],  # Point 2
            [4, 5],  # Point 3
        ]
    )


@pytest.fixture
def simple_data():
    """Fixture for simple test data"""
    return np.array([[0, 0], [1, 0], [10, 0]])


class TestNode:
    def test_leaf_node_creation(self, sample_data):
        """Test creation of leaf nodes"""
        leaf = Node((0,), 0)  # Using tuple for observations
        assert leaf.get_obs() == (0,)
        assert leaf.height == 0
        assert leaf.is_leaf == True
        assert leaf.left is None
        assert leaf.right is None

    def test_merged_node_creation(self, sample_data):
        """Test creation of merged nodes"""
        left = Node((0,), 0)
        right = Node((1,), 0)
        merged = Node(
            (0, 1), 1.0, left=left, right=right  # Using tuple for observations
        )
        assert merged.get_obs() == (0, 1)
        assert merged.height == 1.0
        assert merged.is_leaf == False
        assert merged.left is left
        assert merged.right is right


class TestLinkageFunctions:
    @pytest.mark.parametrize(
        "linkage_func",
        [single_linkage, complete_linkage, average_linkage, centroid_linkage],
    )
    def test_linkage_functions(self, sample_data, linkage_func):
        """Test all linkage functions"""
        # Use actual data points from sample_data
        cluster1 = sample_data[0:1]  # First point as cluster
        cluster2 = sample_data[1:2]  # Second point as cluster

        distance = linkage_func(cluster1, cluster2)
        assert isinstance(distance, float)
        assert distance >= 0


class TestAgglomerativeClustering:
    @pytest.mark.parametrize(
        "linkage_func",
        [single_linkage, complete_linkage, average_linkage, centroid_linkage],
    )
    def test_clustering_initialization(self, sample_data, linkage_func):
        """Test initialization of clustering"""
        hc = AgglomerativeClustering(sample_data, linkage=linkage_func)
        assert hc.levels == len(sample_data) - 1
        assert len(hc.clusters[0]) == len(sample_data)

    def test_final_cluster(self, sample_data):
        """Test if final cluster contains all points"""
        hc = AgglomerativeClustering(sample_data, linkage=single_linkage)
        final_cluster = hc.clusters[hc.levels][0]
        assert list(sorted(final_cluster.get_obs())) == list(range(len(sample_data)))

    def test_get_clusters_valid(self, simple_data):
        """Test get_clusters method with valid input"""
        hc = AgglomerativeClustering(simple_data, linkage=single_linkage)
        clusters = hc.get_clusters(2)
        assert len(clusters) == 2
        assert all(isinstance(cluster, Node) for cluster in clusters)

    def test_clustering_order(self, simple_data):
        """Test if clustering order is correct for simple case"""
        hc = AgglomerativeClustering(simple_data, linkage=single_linkage)
        level_1_clusters = hc.clusters[1]
        merged_cluster = next(
            cluster for cluster in level_1_clusters if len(cluster.get_obs()) == 2
        )
        assert set(merged_cluster.get_obs()) == {0, 1}

    def test_distance_monotonicity(self, sample_data):
        """Test if merge heights are monotonically increasing"""
        hc = AgglomerativeClustering(sample_data, linkage=single_linkage)
        heights = []
        for level in range(hc.levels + 1):
            for cluster in hc.clusters[level]:
                if not cluster.is_leaf:  # Only check merged clusters
                    heights.append(cluster.height)
        assert all(heights[i] <= heights[i + 1] for i in range(len(heights) - 1))

    def test_cluster_consistency(self, sample_data):
        """Test if clusters maintain consistency across levels"""
        hc = AgglomerativeClustering(sample_data, linkage=single_linkage)
        for level in range(hc.levels + 1):
            points_in_level = []
            for cluster in hc.clusters[level]:
                points_in_level.extend(cluster.get_obs())
            assert sorted(points_in_level) == list(range(len(sample_data)))

    @pytest.mark.parametrize("n_clusters", [2, 3])
    def test_get_clusters_output(self, sample_data, n_clusters):
        """Test get_clusters output structure"""
        hc = AgglomerativeClustering(sample_data, linkage=single_linkage)
        clusters = hc.get_clusters(n_clusters)

        assert len(clusters) == n_clusters
        assert all(isinstance(cluster, Node) for cluster in clusters)

        all_points = []
        for cluster in clusters:
            all_points.extend(cluster.get_obs())
        assert len(all_points) == len(sample_data)
        assert len(set(all_points)) == len(sample_data)
