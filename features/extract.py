import scipy.io
import numpy as np
import matplotlib.colors as mc
import cv2
import os

import sift as sift
import features.descriptor as desc


def extract(image_files, image_path, feature_path, descriptor_path):
    """ Extracts feature histogram vectors of classified SIFT features, 
        SIFT location colors and random Gaussian distributed colors 
        for the images.

    :param image_files: A list of image file paths.
    :param image_path: Path to image directory.
    :param feature_path: Path to feature directory.
    :param descriptor_path: Path to descriptor directory.

    :return D: A 2-D array with SIFT descriptor feature histograms for
               each image in rows.
    :return C_desc: A 2-D array with histograms for colors for SIFT
                    descriptor feature locations for each image in rows.
    :return C_rand : A 2-D array with histograms for colors for Gaussian
                     distributed random points for each image in rows.
    """
    
    descriptor_data = scipy.io.loadmat('data/kmeans_descriptor_results.mat')
    color_data = scipy.io.loadmat('data/kmeans_color_results.mat')
    
    # Get descriptor training data cluster centers and descriptor training data 
    # cluster center indexes and create histogram.
    desc_cc = descriptor_data.get('cbest')
    desc_cc_norm = np.histogram(descriptor_data.get('idxbest'), range(1, desc_cc.shape[0] + 2))[0]

    color_cc = color_data.get('ccbest')
    color_cc_norm = np.histogram(color_data.get('idxcbest'), range(1, color_cc.shape[0] + 2))[0]
    
    # Retrieve Gaussian distributed random points for color retrieval.
    gaussians = scipy.io.loadmat('data/gaussians.mat').get('rands')
    x = np.mod((1+gaussians['x'][0,0]/2.3263)/2, 1)[:, 0]
    y = np.mod((1+gaussians['y'][0,0]/2.3263)/2, 1)[:, 0]

    # Extract descriptors and colors for all images
    for image_file in image_files:
        
        image = cv2.imread(os.path.join(image_path, image_file), cv2.CV_LOAD_IMAGE_UNCHANGED)

        if len(image.shape) == 3:
            image = image[:, :, ::-1]  # Reverse to RGB if color image

        im = np.array(image) / 255.0
        shape = im.shape
         
        locs, descs = sift.load_features(feature_path, image_file)
        
        descs = desc.normalize(descs)
        desc_cc = desc.normalize(desc_cc)
        desc_hist = desc.classify_cosine(descs, desc_cc)

        # SIFT descriptors
        norm_desc_hist = desc.normalize_by_division(desc_hist, desc_cc_norm)
        
        # Colors in descriptor locations
        colors_desc_hist = get_color_hist(im, locs[:, 1], locs[:, 0], color_cc, color_cc_norm)
        
        # Colors in Gaussian distributed points.   
        colors_rand_hist = get_color_hist(im, shape[0]*np.array(y), shape[1]*np.array(x), color_cc, color_cc_norm)

        save_descriptor(descriptor_path, image_file, norm_desc_hist, colors_desc_hist, colors_rand_hist)

        print 'Processed {0}'.format(image_file)

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
        return create_NaN_array(cluster_centers.shape[0])


def set_nan_rows_to_mean(X):
    """ Sets rows of a 2-D array with NaN values to 
        the mean of the non NaN values for each column.

    :param X: A 2-D array of row vectors.

    :return A 2-D array where the row vectors with NaN values have been
            changed to the mean of the rest of the row vectors for each column.
    """
    
    C_norm = np.linalg.norm(X, axis=1)
        
    C_real = np.mean(X[~np.isnan(C_norm), :], axis=0)
    C_real = C_real / np.linalg.norm(C_real)
    
    # Set the NaN rows to the mean.    
    X[np.isnan(C_norm), :] = np.tile(C_real, (sum(np.isnan(C_norm)), 1))
    
    return X


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


def create_NaN_array(cols):
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


def save_descriptor(file_path, image, descriptors, descriptor_colors, random_colors):
    """ Saves descriptors to .npz.

    :param file_path: The folder.
    :param image: The image name.
    :param descriptors: Descriptor histogram.
    :param descriptor_colors: Histogram for colors in descriptor locations.
    :param random_colors: Histogram for colors in random locations.
    """

    np.savez(os.path.join(file_path, image + '.descriptors.npz'),
             descriptors=descriptors,
             descriptor_colors=descriptor_colors,
             random_colors=random_colors)


def load_descriptors(file_path, images):
    """ Loads descriptors from .npz. Descriptor color values for grayscale images are set to
        mean of values for RGB images.

    :param file_path: The folder.
    :param images: The image names.

    :return descriptors: Descriptor histograms for all images in rows.
    :return descriptor_colors: Histogram for colors in descriptor locations for all images in rows.
    :return random_colors: Histogram for colors in random locations for all images in rows.
    """

    descriptors = []
    descriptor_colors = []
    random_colors = []

    for image in images:
        d, dc, rc = load_descriptor(file_path, image)

        descriptors.append(d)
        descriptor_colors.append(dc)
        random_colors.append(rc)

    descriptors = np.array(descriptors)
    descriptor_colors = np.array(descriptor_colors)
    random_colors = np.array(random_colors)

    # Set colors for grayscale images to mean of other feature vectors.
    descriptor_colors = set_nan_rows_to_mean(descriptor_colors)
    random_colors = set_nan_rows_to_mean(random_colors)

    return descriptors, descriptor_colors, random_colors


def load_descriptor(file_path, image):
    """ Loads descriptors from .npz.

    :param file_path: The folder.
    :param image: The image name.

    :return descriptors: Descriptor histogram.
    :return descriptor_colors: Histogram for colors in descriptor locations.
    :return random_colors: Histogram for colors in random locations.
    """

    d = np.load(os.path.join(file_path, image + '.descriptors.npz'))

    return d['descriptors'], d['descriptor_colors'], d['random_colors']