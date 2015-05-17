import numpy as np

import features.extract as extract
import pca


def process(features, neutral_factor):
    """ Performs PCA on feature vectors by subtracting a neutral vector.

    :param features: The feature vectors.
    :param neutral_factor: The factor of the neutral vector to subtract.
    :return: Principal component projections of feature vectors.
    """

    N = extract.create_neutral_vector(np.array([[features.shape[1], 1]]), features.shape[0])
    F = features

    Y, V = pca.neutral_sub_pca_vector(F, neutral_factor*N)

    return Y


def process_combined(descriptors, descriptor_colors, random_colors, descriptor_weight, neutral_factor):
    """ Performs PCA on feature vectors by subtracting a neutral vector. The feature vectors are combined
        using the supplied weight.

    :param descriptors:
    :param descriptor_colors:
    :param random_colors:
    :param descriptor_weight: The weight of the descriptors as part of the norm.
    :param neutral_factor: The factor of the neutral vector to subtract.
    :return: Principal component projections of feature vectors.
    """

    color_weight = (1-descriptor_weight)/2

    N = extract.create_neutral_vector(
        np.array([[descriptors.shape[1], np.sqrt(descriptor_weight)],
                  [descriptor_colors.shape[1], np.sqrt(color_weight)],
                  [random_colors.shape[1], np.sqrt(color_weight)]]),
        descriptors.shape[0])
    F = np.hstack((np.sqrt(descriptor_weight)*descriptors,
                   np.hstack((np.sqrt(color_weight)*descriptor_colors, np.sqrt(color_weight)*random_colors))))

    Y, V = pca.neutral_sub_pca_vector(F, neutral_factor*N)

    return Y