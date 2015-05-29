import numpy as np
import os.path
import cv2

import process


def all_structures(data):
    """ Calculates all structures for the principal component projections. Runs k-means a number of times and
        saves the result with lowest distortion.

    :param data: Data set.
    """

    images, pc_projections, pcs = data.pca.load()
    pc_projections_truncated = pc_projections[:, :data.analysis.config.pc_projection_count]

    termination_criteria = (cv2.TERM_CRITERIA_EPS, data.analysis.config.iterations, 0.0001)
    ret, labels, centroids = cv2.kmeans(pc_projections_truncated.astype(np.float32), data.analysis.config.clusters,
                                        termination_criteria, data.analysis.config.runs, cv2.KMEANS_RANDOM_CENTERS)

    structure_indices = []
    for label in range(0, centroids.shape[0]):
        structure_indices.append(np.where(labels == label)[0])

    structure_indices = np.array(structure_indices)

    save_structures(data.analysis.path, centroids, structure_indices)


def score_structures(data):
    """ Scores structures based on the representative result.

    :param data: Data set.
    """

    closest_group, representative = process.load_closest(data.analysis.path)
    centroids, structures = load_structures(data.analysis.path)

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

    save_scored_structures(data.analysis.path, scored_structures)


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