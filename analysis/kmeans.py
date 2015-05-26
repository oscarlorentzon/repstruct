import numpy as np
from scipy.cluster.vq import kmeans2
import os.path

import process


def structures(data, clusters=8):

    images, pc_projections, pcs = process.load_principal_components(data.result_path)
    pc_projections_truncated = pc_projections[:, :data.config.pc_projection_count]

    centroids, labels = kmeans2(pc_projections_truncated, k=clusters, iter=1000, minit='random')

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