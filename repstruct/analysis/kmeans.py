import numpy as np
import cv2


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

    data.analysis.save_structures(centroids, structure_indices)


def score_structures(data):
    """ Scores structures based on the representative result.

    :param data: Data set.
    """

    closest_group, representative = data.analysis.load_closest()
    centroids, structures = data.analysis.load_structures()

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

    data.analysis.save_scored_structures(scored_structures)