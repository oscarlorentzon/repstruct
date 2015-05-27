import numpy as np
from scipy.cluster.vq import kmeans2
import os.path

import process


def all_structures(data, clusters=8, iterations=100, runs=200):

    images, pc_projections, pcs = process.load_principal_components(data.result_path)
    pc_projections_truncated = pc_projections[:, :data.config.pc_projection_count]

    centroids = None
    labels = None
    distortion = float('inf')
    for run in range(0, runs):
        cs, ls = kmeans2(pc_projections_truncated, k=clusters, iter=iterations, minit='random')

        non_empty_clusters = []
        for i in range(0, clusters):
            if np.sum(ls == i) > 0:
                non_empty_clusters.append(i)

        d = 0
        for label in non_empty_clusters:
            observations = pc_projections_truncated[np.where(ls == label)[0]]

            n = np.linalg.norm(observations - cs[label], axis=1)
            d += np.sum(n)

        if d < distortion:
            centroids = cs[non_empty_clusters]
            labels = ls
            distortion = d

    structure_indices = []
    for label in range(0, clusters):
        structure_indices.append(np.where(labels == label)[0])

    structure_indices = np.array(structure_indices)

    save_structures(data.result_path, centroids, structure_indices)


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