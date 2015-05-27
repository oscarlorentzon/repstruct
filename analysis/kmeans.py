import numpy as np
from scipy.cluster.vq import kmeans2
import os.path
import warnings

import process


def all_structures(data, clusters=8, iterations=200, runs=200):

    images, pc_projections, pcs = process.load_principal_components(data.result_path)
    pc_projections_truncated = pc_projections[:, :data.config.pc_projection_count]

    centroids = None
    labels = None
    distortion = float('inf')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore warning from kmeans2 about empty clusters.

        for run in range(0, runs):
            cs, ls = kmeans2(pc_projections_truncated, k=clusters, iter=iterations, minit='random')
            d, non_empty_clusters = k_means_distortion(pc_projections_truncated, ls, cs)

            if d < distortion:
                centroids = cs[non_empty_clusters]
                labels = ls
                distortion = d

    structure_indices = []
    for label in range(0, clusters):
        structure_indices.append(np.where(labels == label)[0])

    structure_indices = np.array(structure_indices)

    save_structures(data.result_path, centroids, structure_indices)


def k_means_distortion(observations, labels, centroids):
    non_empty_clusters = []
    for label in range(0, centroids.shape[0]):
        if np.sum(labels == label) > 0:
            non_empty_clusters.append(label)

    distortion = 0
    for label in non_empty_clusters:
        cluster_observations = observations[np.where(labels == label)[0]]

        n = np.linalg.norm(cluster_observations - centroids[label], axis=1)
        distortion += np.sum(n)

    return distortion, non_empty_clusters


def save_structures(file_path, centroids, structures):
    """ Saves result to .npz.

    :param file_path: The results folder.
    :param centroids:
    :param structures:
    """

    np.savez(os.path.join(file_path, 'structures.npz'),
             centroids=centroids,
             structures=structures)


def load_structures(file_path):
    """ Loads result from .npz.

    :param file_path: The result folder.

    :return centroids:
    :return structures:
    """

    s = np.load(os.path.join(file_path, 'structures.npz'))

    return s['centroids'], s['structures']