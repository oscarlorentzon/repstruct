import scipy.io
import numpy as np
import matplotlib.colors as mc
import os.path as op
import cv2
import os
import multiprocessing

import descriptor as desc


class DescriptorExtractor:

    def __init__(self, feature_data, descriptor_data, collection_data,
                 desc_cc, desc_cc_norm, color_cc, color_cc_norm, x, y):
        """ Creates a descriptor extractor.

        :param feature_data: Feature data set.
        :param descriptor_data: Descriptor data set.
        :param collection_data: Collection data set.
        :param desc_cc: Descriptor cluster centers.
        :param desc_cc_norm: Descriptor cluster center normalization histogram..
        :param color_cc: Color cluster centers.
        :param color_cc_norm: Color cluster center normalization histogram..
        :param x: Gaussian randomly distributed x values.
        :param y: Gaussian randomly distributed y values.
        """

        self.__feature_data = feature_data
        self.__descriptor_data = descriptor_data
        self.__collection_data = collection_data

        self.__desc_cc = desc_cc
        self.__desc_cc_norm = desc_cc_norm

        self.__color_cc = color_cc
        self.__color_cc_norm = color_cc_norm

        self.__x = x
        self.__y = y

    def __call__(self, image_file):
        """ Extracts descriptor, descriptor location color and random location color histograms and saves to .npz.

        :param image_file: Image name.
        """

        image = cv2.imread(os.path.join(self.__collection_data.path, image_file), cv2.CV_LOAD_IMAGE_UNCHANGED)

        if len(image.shape) == 3:
            image = image[:, :, ::-1]  # Reverse to RGB if color image

        im = np.array(image) / 255.0
        shape = im.shape

        locs, descs = self.__feature_data.load(image_file)

        descs = desc.normalize(descs)
        desc_cc = desc.normalize(self.__desc_cc)
        desc_hist = desc.classify_cosine(descs, desc_cc)

        # SIFT descriptors
        norm_desc_hist = desc.normalize_by_division(desc_hist, self.__desc_cc_norm)

        # Colors in descriptor locations
        colors_desc_hist = get_color_hist(im, locs[:, 1], locs[:, 0], self.__color_cc, self.__color_cc_norm)

        # Colors in Gaussian distributed points.
        colors_rand_hist = get_color_hist(im, shape[0]*np.array(self.__y), shape[1]*np.array(self.__x),
                                          self.__color_cc, self.__color_cc_norm)

        self.__descriptor_data.save(image_file, norm_desc_hist, colors_desc_hist, colors_rand_hist)

        print 'Processed {0}'.format(image_file)


def extract(data):
    """ Extracts feature histogram vectors of classified SIFT features, 
        SIFT location colors and random Gaussian distributed colors 
        for the images.

    :param data: Data set.

    :return D: A 2-D array with SIFT descriptor feature histograms for
               each image in rows.
    :return C_desc: A 2-D array with histograms for colors for SIFT
                    descriptor feature locations for each image in rows.
    :return C_rand : A 2-D array with histograms for colors for Gaussian
                     distributed random points for each image in rows.
    """

    data_folder = op.abspath(op.join(op.dirname(__file__), 'data'))

    descriptor_mat = scipy.io.loadmat(op.join(data_folder, 'kmeans_descriptor_results.mat'))
    color_mat = scipy.io.loadmat(op.join(data_folder, 'kmeans_color_results.mat'))
    
    # Get descriptor training data cluster centers and descriptor training data 
    # cluster center indexes and create histogram.
    desc_cc = descriptor_mat.get('cbest')
    desc_cc_norm = np.histogram(descriptor_mat.get('idxbest'), range(1, desc_cc.shape[0] + 2))[0]

    color_cc = color_mat.get('ccbest')
    color_cc_norm = np.histogram(color_mat.get('idxcbest'), range(1, color_cc.shape[0] + 2))[0]
    
    # Retrieve Gaussian distributed random points for color retrieval.
    gaussians = scipy.io.loadmat(op.join(data_folder, 'gaussians.mat')).get('rands')
    x = np.mod((1+gaussians['x'][0, 0]/2.3263)/2, 1)[:, 0]
    y = np.mod((1+gaussians['y'][0, 0]/2.3263)/2, 1)[:, 0]

    descriptor_extractor = DescriptorExtractor(data.feature, data.descriptor, data.collection,
                                               desc_cc, desc_cc_norm, color_cc, color_cc_norm, x, y)

    if data.collection.config.processes == 1:
        for image_file in data.collection.images():
            descriptor_extractor(image_file)
    else:
        pool = multiprocessing.Pool(data.collection.config.processes)
        pool.map(descriptor_extractor, data.collection.images())

    print 'Images processed'


def get_color_hist(image, rows, columns, cluster_centers, cluster_center_norm):
    """ Creates a normalized histogram of for the colors of the rows and 
        columns of an image when classified against the cluster centers on the
        HS disc of the HSV color space and divided with the cluster center norm.

    :param image: A 3-D array of RGB values.
    :param rows: The row points of the image
    :param columns: The column points of the image.
    :param cluster_centers: The cluster centers.
    :param cluster_center_norm: The cluster center norm.

    :return A 2-D array where the row vectors with NaN values have been changed to the mean
            of the rest of the row vectors for each column.
    """
    
    if len(image.shape) == 3:             
        rgb = get_rgb_from_locs(rows, columns, image)
        hs_coords = rgb_to_hs_coords(rgb)
        hist = desc.classify_euclidean(hs_coords, cluster_centers)
        return desc.normalize_by_division(hist, cluster_center_norm)
    else:
        return create_nan_array(cluster_centers.shape[0])


def create_nan_array(cols):
    """ Creates a 2-D array with NaN values.

     :param rows: The number of rows.
     :param cols: The number of columns.

     :return A 2-D array with only NaN values.
    """
    
    nan_array = np.empty((cols))
    nan_array[:] = np.NAN
    return nan_array


def rgb_to_hs_coords(rgb):
    """ Converts RGB values to x and y coordinates on the HSV
        disc irrespective of the value component.

    :param rgb: (..., 3) array-like. All values must be in the range [0, 1]

    :return coords: (..., 2) ndarray Colors converted to x and y
            coordinates on the HS disc of the HS color space in the in range [0, 1].
    """
    
    hsv = mc.rgb_to_hsv(rgb)
    
    x = np.multiply(hsv[:, 0, 1], np.cos(2*np.pi*hsv[:, 0, 0]))
    y = np.multiply(hsv[:, 0, 1], np.sin(2*np.pi*hsv[:, 0, 0]))
    
    return np.vstack((x, y)).transpose()


def get_rgb_from_locs(locs_r, locs_c, im):
    """ Retrieves the RGB values for an image in the specified locations.

    :param locs_r: An array with the rows of the image locations.
    :param locs_c: An array with the columns of the image locations.
    :param im: Image array.

    :return rgb: (..., 2) ndarray The RGB values in the specified locations.
    """
    
    locs_row = np.empty((locs_r.shape[0],), dtype='int')
    locs_col = np.empty((locs_r.shape[0],), dtype='int')
    np.floor(locs_r, out=locs_row)
    np.floor(locs_c, out=locs_col)

    rgb = im[locs_row, locs_col]

    return np.resize(rgb, (rgb.shape[0], 1, rgb.shape[1]))