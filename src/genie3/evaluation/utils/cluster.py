"""
Hierarchical clustering utilities for the Genie pipeline.

Provides TM-score-based hierarchical clustering of protein structures
using single, complete, or average linkage.
"""

from typing import List
import numpy as np


def hcluster(
    dists: np.ndarray,
    linkage: str,
    max_ctm_threshold: float,
) -> List[List[int]]:
    """
    Perform hierarchical clustering based on pairwise TM scores.

    Merges the two closest clusters (by TM score) at each step until no
    remaining pair exceeds ``max_ctm_threshold``.

    Args:
        dists: [N, N] pairwise TM-score matrix (higher = more similar).
        linkage: Linkage method — one of 'single', 'complete', or 'average'.
        max_ctm_threshold: Minimum inter-cluster TM score required to merge.

    Returns:
        List[List[int]]: Each inner list contains the indices of one cluster.
    """

    def compute_cluster_tm(cluster_i: List[int], cluster_j: List[int]) -> float:
        """
        Compute the TM-score distance between two clusters.

        Uses the symmetric TM score min(dists[i][j], dists[j][i]) for each
        pair, then aggregates according to the chosen linkage method.

        Args:
            cluster_i: Indices belonging to the first cluster.
            cluster_j: Indices belonging to the second cluster.

        Returns:
            float: Inter-cluster TM score (higher = more similar).
        """
        scores = [
            min(dists[i][j], dists[j][i])
            for i in cluster_i
            for j in cluster_j
        ]
        if linkage == "single":
            return max(scores)      # closest pair
        elif linkage == "complete":
            return min(scores)      # farthest pair
        else:                       # average
            return sum(scores) / len(scores)

    # Initialize: each structure is its own cluster
    clusters = [[i] for i in range(dists.shape[0])]

    while len(clusters) > 1:
        # Find the two most similar clusters
        best_i, best_j, best_ctm = None, None, 0.0
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                ctm = compute_cluster_tm(clusters[i], clusters[j])
                if ctm > best_ctm:
                    best_i, best_j, best_ctm = i, j, ctm

        # Stop if no pair is similar enough to merge
        if best_ctm < max_ctm_threshold:
            break

        # Merge the two closest clusters (remove higher index first to preserve indices)
        new_cluster = clusters[best_i] + clusters[best_j]
        del clusters[best_j]
        del clusters[best_i]
        clusters.append(new_cluster)

    return clusters