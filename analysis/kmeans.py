import numpy as np
from scipy.cluster.vq import kmeans2
import os.path
import warnings

import process


def all_structures(data, clusters=8, iterations=100, runs=200):

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


def score_structures(data):
    closest_group, representative = process.load_closest(data.result_path)
    centroids, structures = load_structures(data.result_path)

    scores = []
    lengths = []
    for structure in structures:
        lengths.append(len(structure))

        score = 0
        for index in structure:
            if index in representative:
                score += 3
            elif index in closest_group:
                score += 1

        scores.append(score)

    scores = np.array(scores)
    lengths = np.array(lengths)

    # When multiple clusters have the same score the one with most images is ranked higher.
    length_scores = np.max(lengths) * scores + lengths

    ordered = np.argsort(length_scores)[::-1]  # Sort and reverse to get descending
    scored_structures = structures[ordered]

    save_scored_structures(data.result_path, scored_structures)


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


def save_scored_structures(file_path, scored_structures):
    """ Saves result to .npz.

    :param file_path: The results folder.
    :param scored_structures:
    """

    np.savez(os.path.join(file_path, 'scored_structures.npz'), scored_structures=scored_structures)


def load_scored_structures(file_path):
    """ Loads result from .npz.

    :param file_path: The result folder.

    :return scored_structures:
    """

    s = np.load(os.path.join(file_path, 'scored_structures.npz'))

    return s['scored_structures']