import os
import cv2
import numpy as np

from multiprocessing import Pool


class SiftExtractor:

    def __init__(self, image_path, feature_path):
        """ Creates a SiftExtractor.

            Parameters
            ----------
            image_path: Path to image files.
            feature_path: Path to feature files.
        """

        self.image_path = image_path
        self.feature_path = feature_path

    def __call__(self, image):
        """ Extracts SIFT features for an image and saves the descriptors and locations to file.

            Parameters
            ----------
            image: Image name.
        """

        im = cv2.imread(os.path.join(self.image_path, image), cv2.IMREAD_GRAYSCALE)
        locs, descs = extract_feature_vectors(im)

        save_features(self.feature_path, image, locs, descs)

        print 'Extracted {0} features for {1}'.format(descs.shape[0], image)


def extract(images, image_path, feature_path, processes=6):
    """ Extracts SIFT features for a list of images. Saves the descriptors and locations to file.

        Parameters
        ----------
        image_files: Image names.
        image_path: Folder path for image files.
        feature_path: Folder path for features.
    """
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)

    sift_extractor = SiftExtractor(image_path, feature_path)
    if processes == 1:
        for image in images:
            sift_extractor(image)
    else:
        pool = Pool(processes)
        pool.map(sift_extractor, images)

    print 'Features extracted'


def extract_feature_vectors(image, edge_threshold=10, peak_threshold=0.001):
    """ Process a grayscale image and return the found SIFT feature points and descriptors.

        Parameters
        ----------
        image : A gray scale image represented in a 2-D array.
        edge_threshold: The edge threshold.
        peak_threshold: The peak threshold.

        Returns
        -------
        locs : An array with the row, column, scale and orientation of each feature.
        descs : The descriptors.
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

        Parameters
        ----------
        file_path: The folder.
        image: The image name.
        locations: Descriptor locations.
        descriptors: Descriptor vectors.
    """

    np.savez(os.path.join(file_path, image + '.npz'), locations=locations, descriptors=descriptors)


def load_features(file_path, image):
    """ Loads features from .npz.

        Parameters
        ----------
        file_path: The folder.
        image: The image name.
    """

    f = np.load(os.path.join(file_path, image + '.npz'))

    return f['locations'], f['descriptors']