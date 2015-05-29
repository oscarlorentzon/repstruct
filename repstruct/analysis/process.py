import numpy as np
import os

from repstruct.featuremode import FeatureMode
import repstruct.features.extract as extract
import pca
import kclosest


def process_features(features, neutral_factor):
    """ Performs PCA on feature vectors by subtracting a neutral vector.

    :param features: The feature vectors.
    :param neutral_factor: The factor of the neutral vector to subtract.
    :return: Principal component projections of feature vectors.
    :return: Principal components.
    """

    N = create_neutral_vector(np.array([[features.shape[1], 1]]), features.shape[0])
    F = features

    pc_projections, pcs = pca.neutral_sub_pca_vector(F, neutral_factor*N)

    return pc_projections, pcs


def process_combined_features(descriptors, descriptor_colors, random_colors, descriptor_weight, neutral_factor):
    """ Performs PCA on feature vectors by subtracting a neutral vector. The feature vectors are combined
        using the supplied weight.

    :param descriptors:
    :param descriptor_colors:
    :param random_colors:
    :param descriptor_weight: The weight of the descriptors as part of the norm.
    :param neutral_factor: The factor of the neutral vector to subtract.
    :return: Principal component projections of feature vectors.
    :return: Principal components.
    """

    color_weight = (1-descriptor_weight)/2

    N = create_neutral_vector(
        np.array([[descriptors.shape[1], np.sqrt(descriptor_weight)],
                  [descriptor_colors.shape[1], np.sqrt(color_weight)],
                  [random_colors.shape[1], np.sqrt(color_weight)]]),
        descriptors.shape[0])
    F = np.hstack((np.sqrt(descriptor_weight)*descriptors,
                   np.hstack((np.sqrt(color_weight)*descriptor_colors, np.sqrt(color_weight)*random_colors))))

    pc_projections, pcs = pca.neutral_sub_pca_vector(F, neutral_factor*N)

    return pc_projections, pcs


def process(data):
    """ Processes feature vectors according to feature mode specified in data set. Saves result to file.

    :param data: Data set with feature mode, neutral factor and descriptor weight.
    """

    images = data.images()
    descriptors, descriptor_colors, random_colors = extract.load_descriptors(data.descriptor, images)

    if data.config.feature_mode == FeatureMode.Colors:
        pc_projections, pcs = process_features(random_colors, data.config.neutral_factor)
    elif data.config.feature_mode == FeatureMode.Descriptors:
        pc_projections, pcs = process_features(descriptors, data.config.neutral_factor)
    else:
        pc_projections, pcs = process_combined_features(descriptors, descriptor_colors, random_colors,
                                         data.config.descriptor_weight, data.config.neutral_factor)

    save_principal_components(data.result_path, images, pc_projections, pcs)


def closest(data):
    """ Determines the closest group and the most representative images and saves to file. Loads image list
        and corresponding principal components from file.

    :param data: Data set.
    """

    images, pc_projections, pcs = load_principal_components(data.result_path)

    pc_projections_truncated = pc_projections[:, :data.config.pc_projection_count]

    closest_group_count = int(round(data.config.closest_group * images.shape[0], 0))
    representative_count = int(round(data.config.representative * images.shape[0], 0))

    closest_group = kclosest.k_closest(closest_group_count, pc_projections_truncated)
    representative = closest_group[kclosest.k_closest(representative_count, pc_projections_truncated[closest_group, :])]

    save_closest(data.result_path, closest_group, representative)


def create_neutral_vector(D, rows):
    """ Creates a 2-D array with neutral vectors according to the
        size and weights specified in a 2-D array.

    The neutral vector rows is only normalized if the input
    parameters are weighted correctly.

    :param D: 2-D array with rows specifying the length and weight
              for each section of the neutral vector.
    :param rows: An integer specifying the number of rows in the
                 neutral 2-D array.

    :return A 2-D array with rows with values according to the length
            and weight requirements in the input.
    """

    N = np.array([]).reshape(rows, 0)

    for d in D:
        N = np.concatenate((N, d[1]*np.sqrt(1.0/d[0])*np.array([np.ones(d[0]),]*rows)), axis=1)

    return N


def save_principal_components(file_path, images, pc_projections, principal_components):
    """ Saves result to .npz.

    :param file_path: The results folder.
    :param images: The image names.
    :param pc_projections: The principal component projection arrays.
    :param principal_components: The principal components.
    """

    np.savez(os.path.join(file_path, 'principal_components.npz'),
             images=images,
             pc_projections=pc_projections,
             principal_components=principal_components)


def load_principal_components(file_path):
    """ Loads principal components from .npz.

    :param file_path: The results folder.

    :return images: The image names.
    :return pc_projections: The principal component projection arrays.
    :return principal_components: The principal components.
    """

    p = np.load(os.path.join(file_path, 'principal_components.npz'))

    return p['images'], p['pc_projections'], p['principal_components']


def save_closest(file_path, closest_group, representative):
    """ Saves result to .npz.

    :param file_path: The results folder.
    :param closest_group: The image indices of the closest group.
    :param representative: The image indices of the representative group.
    """

    np.savez(os.path.join(file_path, 'closest.npz'),
             closest_group=closest_group,
             representative=representative)


def load_closest(file_path):
    """ Loads result from .npz.

    :param file_path: The result folder.

    :return closest_group: The image indices of the closest group.
    :return representative: The image indices of the representative group.
    """

    c = np.load(os.path.join(file_path, 'closest.npz'))

    return c['closest_group'], c['representative']


