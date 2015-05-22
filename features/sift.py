import os
import cv2
import numpy as np

from multiprocessing import Pool


class SiftExtractor:

    def __init__(self, image_path, feature_path, edge_threshold, peak_threshold):
        """ Creates a SiftExtractor.

        :param image_path: Path to image files.
        :param feature_path: Path to feature files.
        :param edge_threshold: SIFT edge threshold.
        :param peak_threshold: SIFT peak threshold.
        """

        self.__image_path = image_path
        self.__feature_path = feature_path
        self.__edge_threshold = edge_threshold
        self.__peak_threshold = peak_threshold

    def __call__(self, image):
        """ Extracts SIFT features for an image and saves the descriptors
            and locations to file.

        :param image: Image name.
        """

        im = cv2.imread(os.path.join(self.__image_path, image), cv2.IMREAD_GRAYSCALE)
        locations, descriptors = extract_feature_vectors(im, self.__edge_threshold, self.__peak_threshold)

        save_features(self.__feature_path, image, locations, descriptors)

        print 'Extracted {0} features for {1}'.format(descriptors.shape[0], image)


def extract(data):
    """ Extracts SIFT features for a list of images. Saves the descriptors
        and locations to file.

    :param data: Data set.
    """

    sift_extractor = SiftExtractor(data.image_path, data.feature_path,
                                   data.config.edge_threshold, data.config.peak_threshold)
    if data.config.processes == 1:
        for image in data.images():
            sift_extractor(image)
    else:
        pool = Pool(data.config.processes)
        pool.map(sift_extractor, data.images())

    print 'Features extracted'


def extract_feature_vectors(image, edge_threshold=10, peak_threshold=0.001):
    """ Process a grayscale image and return the found SIFT feature points
        and descriptors.

    :param image : A gray scale image represented in a 2-D array.
    :param edge_threshold: The edge threshold.
    :param peak_threshold: The peak threshold.

    :return locs : An array with the row, column, scale and orientation of each feature.
    :return descs : The descriptors.
    """

    detector = cv2.FeatureDetector_create('SIFT')
    descriptor = cv2.DescriptorExtractor_create('SIFT')
    detector.setDouble('edgeThreshold', edge_threshold)
    detector.setDouble("contrastThreshold", peak_threshold)

    locs = detector.detect(image)
    locs, desc = descriptor.compute(image, locs)

    locs = np.array([(i.pt[0], i.pt[1], i.size, i.angle) for i in locs])

    return locs, desc


def save_features(file_path, image, locations, descriptors):
    """ Saves features to .npz.

    :param file_path: The folder.
    :param image: The image name.
    :param locations: Descriptor locations.
    :param descriptors: Descriptor vectors.
    """

    np.savez(os.path.join(file_path, image + '.sift.npz'), locations=locations, descriptors=descriptors)


def load_features(file_path, image):
    """ Loads features from .npz.

    :param file_path: The folder.
    :param image: The image name.

    :return Feature locations and feature descriptors.
    """

    f = np.load(os.path.join(file_path, image + '.sift.npz'))

    return f['locations'], f['descriptors']