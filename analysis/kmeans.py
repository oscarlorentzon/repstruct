import numpy as np
from scipy.cluster.vq import kmeans2
import os.path
import warnings

import process


def all_structures(data):
    """ Calculates all structures for the principal component projections. Runs k-means a number of times and
        saves the result with lowest distortion.

    :param data: Data set.
    """

    images, pc_projections, pcs = process.load_principal_components(data.result_path)
    pc_projections_truncated = pc_projections[:, :data.config.pc_projection_count]

    centroids = None
    labels = None
    distortion = float('inf')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore warning from k-means about empty clusters.

        for run in range(0, data.config.runs):
            cs, ls = kmeans2(pc_projections_truncated,
                             k=data.config.clusters, iter=data.config.iterations, minit='random')

            d, non_empty_clusters = k_means_distortion(pc_projections_truncated, ls, cs)

            if d < distortion:
                centroids = cs[non_empty_clusters]
                labels = ls
                distortion = d

    structure_indices = []
    for label in range(0, centroids.shape[0]):
        structure_indices.append(np.where(labels == label)[0])

    structure_indices = np.array(structure_indices)

    save_structures(data.result_path, centroids, structure_indices)


def k_means_distortion(observations, labels, centroids):
    """ Determines the distortion for a k-means run by calculating the total norm of all observations to their cluster
        centroids.

    :param observations: The observations.
    :param labels: The cluster label for each observation.
    :param centroids: The cluster centroids.

    :return distortion: The distortion of the k-means result.
    :return non_empty_clusters: The indices of the non empty clusters.
    """

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
    """ Scores structures based on the representative result.

    :param data: Data set.
    """

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
    :param centroids: The cluster centroids.
    :param structures: Array of structures containing indices for images connected to each cluster centroid.
    """

    np.savez(os.path.join(file_path, 'structures.npz'),
             centroids=centroids,
             structures=structures)


def load_structures(file_path):
    """ Loads result from .npz.

    :param file_path: The result folder.

    :param centroids: The cluster centroids.
    :param structures: Array of structures containing indices for images connected to each cluster centroid.
    """

    s = np.load(os.path.join(file_path, 'structures.npz'))

    return s['centroids'], s['structures']


def save_scored_structures(file_path, scored_structures):
    """ Saves result to .npz.

    :param file_path: The results folder.
    :param scored_structures: Arrays of structures ordered base on score.
    """

    np.savez(os.path.join(file_path, 'scored_structures.npz'), scored_structures=scored_structures)


def load_scored_structures(file_path):
    """ Loads result from .npz.

    :param file_path: The result folder.

    :return scored_structures: Array of structures ordered base on score.
    """

    s = np.load(os.path.join(file_path, 'scored_structures.npz'))

    return s['scored_structures']