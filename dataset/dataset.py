import yaml
import os.path as op
import numpy as np
from os import makedirs, listdir

from features.featuremode import FeatureMode


class DataSet:

    def __init__(self, tag=None, root_path=None):
        """ Creates a data set which holds paths, image list and config values.

        :param tag: Flickr tag.
        :param root_path: Path to directory including config file. Path where tags directory will be created.
        """

        self.__tag = tag

        if root_path is None:
            return

        self.__data_path = root_path + "/tags/" + self.__tag + "/"

        self.__image_path = self.__data_path + "images/"
        self.__feature_path = self.__data_path + "features/"
        self.__descriptor_path = self.__data_path + "descriptors/"
        self.__result_path = self.__data_path + "results/"

        # Load the configuration file and stores values in properties.
        with open(op.join(root_path, 'config.yaml')) as fin:
            self.__config = yaml.load(fin)

        self.__descriptor_weight = self.__config['descriptor_weight']
        self.__neutral_factor = self.__config['neutral_factor']

        feature_mode = self.__config['feature_mode'].upper()
        if feature_mode == 'ALL':
            self.__feature_mode = FeatureMode.All
        elif feature_mode == 'DESCRIPTORS':
            self.__feature_mode = FeatureMode.Descriptors
        elif feature_mode == 'COLORS':
            self.__feature_mode = FeatureMode.Colors
        else:
            raise ValueError('Unknown feature mode (must be ALL, DESCRIPTORS or COLORS)')

        self.__processes = self.__config['processes']

        # Create tag directories if not created.
        for p in [self.__image_path, self.__feature_path, self.__descriptor_path, self.__result_path]:
            if not op.exists(p):
                makedirs(p)

    @property
    def tag(self):
        return self.__tag

    @tag.setter
    def tag(self, tag):
        self.__tag = tag

    @property
    def image_path(self):
        """ The path to the image directory. """
        return self.__image_path

    @image_path.setter
    def image_path(self, image_path):
        self.__image_path = image_path

    @property
    def feature_path(self):
        """ The path to the feature directory. """
        return self.__feature_path

    @feature_path.setter
    def feature_path(self, feature_path):
        self.__feature_path = feature_path

    @property
    def descriptor_path(self):
        """ The path to the descriptor directory. """
        return self.__descriptor_path

    @descriptor_path.setter
    def descriptor_path(self, descriptor_path):
        self.__descriptor_path = descriptor_path

    @property
    def result_path(self):
        """ The path to the result directory. """
        return self.__result_path

    @result_path.setter
    def result_path(self, result_path):
        self.__result_path = result_path

    @property
    def descriptor_weight(self):
        """ The weight of the descriptors with respect to the L2 norm of the combined feature vector. """
        return self.__descriptor_weight

    @descriptor_weight.setter
    def descriptor_weight(self, descriptor_weight):
        self.__descriptor_weight = descriptor_weight

    @property
    def neutral_factor(self):
        """ The factor of the neutral vector to be subtracted in the neutral vector subtraction PCA. """
        return self.__neutral_factor

    @neutral_factor.setter
    def neutral_factor(self, neutral_factor):
        self.__neutral_factor = neutral_factor

    @property
    def feature_mode(self):
        """ Feature mode to be used when processing feature vectors. """
        return self.__feature_mode

    @feature_mode.setter
    def feature_mode(self, feature_mode):
        self.__feature_mode = feature_mode

    @property
    def processes(self):
        """ Number of parallel processes for downloading and extraction. """
        return self.__processes

    @processes.setter
    def processes(self, processes):
        self.__processes = processes

    def images(self):
        """ Lists all images in the image directory.

        :return: List of image names.
        """

        return np.array([im for im in listdir(self.image_path)
                         if op.isfile(op.join(self.image_path, im)) and im.endswith(".jpg")])