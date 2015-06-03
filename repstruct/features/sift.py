import os
import cv2
import numpy as np

import multiprocessing


class SiftExtractor:

    def __init__(self, feature_data, collection_data):
        """ Creates a SiftExtractor.

        :param feature_data: Feature data set.
        :param collection_data: Collection data set.
        """

        self.__feature_data = feature_data
        self.__collection_data = collection_data

    def __call__(self, image):
        """ Extracts SIFT features for an image and saves the descriptors
            and locations to file.

        :param image: Image name.
        """

        im = cv2.imread(os.path.join(self.__collection_data.path, image), cv2.IMREAD_GRAYSCALE)

        if im is None:
            # Image is corrupt. Delete the image file.
            print 'Removing corrupt image {0}'.format(image)
            os.remove(os.path.join(self.__collection_data.path, image))
            return

        locations, descriptors = extract_sift_features(im, self.__feature_data.config.edge_threshold,
                                                       self.__feature_data.config.peak_threshold)

        self.__feature_data.save(image, locations, descriptors)

        print 'Extracted {0} features for {1}'.format(descriptors.shape[0], image)


def extract(data):
    """ Extracts SIFT features for a list of images. Saves the descriptors
        and locations to file.

    :param data: Data set.
    """

    sift_extractor = SiftExtractor(data.feature, data.collection)
    if data.collection.config.processes == 1:
        for image in data.collection.images():
            sift_extractor(image)
    else:
        pool = multiprocessing.Pool(data.collection.config.processes)
        pool.map(sift_extractor, data.collection.images())

    print 'Features extracted'


def extract_sift_features(image, edge_threshold=10, peak_threshold=0.01):
    """ Process a grayscale image and return the found SIFT feature points
        and descriptors.

    :param image : A gray scale image represented in a 2-D array.
    :param edge_threshold: The edge threshold.
    :param peak_threshold: The peak threshold.

    :return locations : An array with the row, column, scale and orientation of each feature.
    :return descriptors : The descriptors.
    """

    detector = cv2.FeatureDetector_create('SIFT')
    descriptor = cv2.DescriptorExtractor_create('SIFT')
    detector.setDouble('edgeThreshold', edge_threshold)
    detector.setDouble("contrastThreshold", peak_threshold)

    locations = detector.detect(image)
    locations, descriptors = descriptor.compute(image, locations)

    locations = np.array([(loc.pt[0], loc.pt[1], loc.size, loc.angle) for loc in locations])

    return locations, descriptors