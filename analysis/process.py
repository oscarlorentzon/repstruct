import numpy as np
import os

from features.featuremode import FeatureMode
import features.extract as extract
import pca
import kclosest


def process_features(features, neutral_factor):
    """ Performs PCA on feature vectors by subtracting a neutral vector.

    :param features: The feature vectors.
    :param neutral_factor: The factor of the neutral vector to subtract.
    :return: Principal component projections of feature vectors.
    """

    N = extract.create_neutral_vector(np.array([[features.shape[1], 1]]), features.shape[0])
    F = features

    Y, V = pca.neutral_sub_pca_vector(F, neutral_factor*N)

    return Y


def process_combined_features(descriptors, descriptor_colors, random_colors, descriptor_weight, neutral_factor):
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


def process(data, principal_component_count=30, closest_group_count=30, representative_count=5):
    """ Processes feature vectors according to feature mode specified in data set. Saves result to file.

    :param data: Data set with feature mode.
    :param principal_component_count: Number of principal component projections to use in closest group estimation.
    :param closest_group_count: Number of images to use for closest group calculation.
    :param representative_count: Number images to use for representative group.
    """

    images = data.images()

    descriptors, descriptor_colors, random_colors = extract.load_descriptors(data.descriptor_path, images)

    if data.feature_mode == FeatureMode.Colors:
        Y = process_features(random_colors, data.neutral_factor)
    elif data.feature_mode == FeatureMode.Descriptors:
        Y = process_features(descriptors, data.neutral_factor)
    else:
        Y = process_combined_features(descriptors, descriptor_colors, random_colors,
                                      data.descriptor_weight, data.neutral_factor)

    Y_truncated = Y[:, :principal_component_count]
    closest_group = kclosest.k_closest(closest_group_count, Y_truncated)
    representative = closest_group[kclosest.k_closest(representative_count, Y_truncated[closest_group, :])]

    save_result(data.result_path, images, Y, closest_group, representative)


def save_result(file_path, images, pc_projections, closest_group, representative):
    """ Saves result to .npz.

    :param file_path: The results folder.
    :param images: The image names.
    :param principal_components: The principal component arrays.
    :param closest_group: The image indices of the closest group.
    :param representative: The image indices of the representative group.
    """

    np.savez(os.path.join(file_path, 'results.npz'),
             images=images,
             pc_projections=pc_projections,
             closest_group=closest_group,
             representative=representative)


def load_result(file_path):
    """ Loads result from .npz.

    :param file_path: The result folder.

    :return  images: The image names.
    :return pc_projections: The principal component projection arrays.
    :return closest_group: The image indices of the closest group.
    :return representative: The image indices of the representative group.
    """

    r = np.load(os.path.join(file_path, 'results.npz'))

    return r['images'], r['pc_projections'], r['closest_group'], r['representative']