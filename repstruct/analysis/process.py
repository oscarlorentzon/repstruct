import numpy as np

from repstruct.configuration import FeatureMode
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

    pc_projections, pcs = pca.neutral_sub_pca_vector(features, neutral_factor*N)

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
        np.array([[descriptors.shape[1], descriptor_weight],
                  [descriptor_colors.shape[1], color_weight],
                  [random_colors.shape[1], color_weight]]),
        descriptors.shape[0])
    F = np.hstack((np.sqrt(descriptor_weight)*descriptors,
                   np.hstack((np.sqrt(color_weight)*descriptor_colors, np.sqrt(color_weight)*random_colors))))

    pc_projections, pcs = pca.neutral_sub_pca_vector(F, neutral_factor*N)

    return pc_projections, pcs


def process(data):
    """ Processes feature vectors according to feature mode specified in data set. Saves result to file.

    :param data: Data set with feature mode, neutral factor and descriptor weight.
    """

    images = data.collection.images()
    descriptors, descriptor_colors, random_colors = load_descriptors(data.descriptor, images)

    if data.pca.config.feature_mode == FeatureMode.Colors:
        pc_projections, pcs = process_features(random_colors, data.pca.config.neutral_factor)
    elif data.pca.config.feature_mode == FeatureMode.Descriptors:
        pc_projections, pcs = process_features(descriptors, data.pca.config.neutral_factor)
    else:
        pc_projections, pcs = process_combined_features(descriptors, descriptor_colors, random_colors,
                                                        data.pca.config.descriptor_weight,
                                                        data.pca.config.neutral_factor)

    data.pca.save(images, pc_projections, pcs)


def closest(data):
    """ Determines the closest group and the most representative images and saves to file. Loads image list
        and corresponding principal components from file.

    :param data: Data set.
    """

    images, pc_projections, pcs = data.pca.load()

    pc_projections_truncated = pc_projections[:, :data.analysis.config.pc_projection_count]

    closest_group_count = int(round(data.analysis.config.closest_group * images.shape[0], 0))
    representative_count = int(round(data.analysis.config.representative * images.shape[0], 0))

    closest_group = kclosest.k_closest(closest_group_count, pc_projections_truncated)
    representative = closest_group[kclosest.k_closest(representative_count, pc_projections_truncated[closest_group, :])]

    data.analysis.save_closest(closest_group, representative)


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

    if not np.abs(np.sum(D[:, 1]) - 1.) < 0.0000001:
        raise AssertionError('Total weight must be 1.')

    N = np.array([]).reshape(1, 0)

    for d in D:
        k = np.sqrt(d[1]/d[0])
        N = np.concatenate((N, k * np.ones((1, d[0]))), axis=1)

    return np.tile(N, (rows, 1))


def load_descriptors(descriptor_data, images):
    """ Loads descriptors from .npz. Descriptor color values for grayscale images are set to
        mean of values for RGB images.

    :param descriptor_data: Descriptor data set.
    :param images: The image names.

    :return descriptors: Descriptor histograms for all images in rows.
    :return descriptor_colors: Histogram for colors in descriptor locations for all images in rows.
    :return random_colors: Histogram for colors in random locations for all images in rows.
    """

    descriptors = []
    descriptor_colors = []
    random_colors = []

    for image in images:
        d, dc, rc = descriptor_data.load(image)

        descriptors.append(d)
        descriptor_colors.append(dc)
        random_colors.append(rc)

    descriptors = np.array(descriptors)
    descriptor_colors = np.array(descriptor_colors)
    random_colors = np.array(random_colors)

    # Set colors for grayscale images to mean of other feature vectors.
    descriptor_colors = set_nan_rows_to_normalized_mean(descriptor_colors)
    random_colors = set_nan_rows_to_normalized_mean(random_colors)

    return descriptors, descriptor_colors, random_colors


def set_nan_rows_to_normalized_mean(X):
    """ Sets rows of a 2-D array with NaN values to
        the mean of the non NaN values for each column
        and normalizes the former NaN rows.

    :param X: A 2-D array of row vectors.

    :return A 2-D array where the row vectors with NaN values have been
            changed to the mean of the rest of the row vectors for each column.
    """

    C_norm = np.linalg.norm(X, axis=1)

    C_real = np.mean(X[~np.isnan(C_norm), :], axis=0)
    C_real = C_real / np.linalg.norm(C_real, axis=0)

    # Set the NaN rows to the mean.
    X[np.isnan(C_norm), :] = np.tile(C_real, (sum(np.isnan(C_norm)), 1))

    return X