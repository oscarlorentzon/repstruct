import os
import cv2
import numpy as np

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