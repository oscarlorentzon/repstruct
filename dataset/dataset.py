import yaml
import os.path as op
import numpy as np
from os import makedirs, listdir

from features.featuremode import FeatureMode


class DataSet:

    def __init__(self, root_path, tag):
        """ Creates a data set which holds paths, image list and config values.
        """

        self.__data_dir = root_path + "/tags/" + tag + "/"

        self.image_dir = self.__data_dir + "images/"
        self.feature_dir = self.__data_dir + "features/"
        self.descriptor_dir = self.__data_dir + "descriptors/"

        self.tag = tag

        self.__load_config(root_path)

        for p in [self.image_dir, self.feature_dir, self.descriptor_dir]:
            if not op.exists(p):
                makedirs(p)

    def __load_config(self, config_path):
        """ Loads the configuration file and stores values in properties.

        :param config_path: The path to the folder containing the configuration file.
        """

        with open(op.join(config_path, 'config.yaml')) as fin:
            self.__config = yaml.load(fin)

        self.descriptor_weight = self.__config['descriptor_weight']
        self.neutral_factor = self.__config['neutral_factor']
        self.processes = self.__config['processes']

        feature_mode = self.__config['feature_mode'].upper()
        if feature_mode == 'ALL':
            self.feature_mode = FeatureMode.All
        elif feature_mode == 'DESCRIPTORS':
            self.feature_mode = FeatureMode.Descriptors
        elif feature_mode == 'COLORS':
            self.feature_mode = FeatureMode.Colors
        else:
            raise ValueError('Unknown feature mode (must be ALL, DESCRIPTORS or COLORS)')

    def images(self):
        """ Lists all images in the image directory.

        :return: List of image names.
        """

        return np.array([im for im in listdir(self.image_dir)
                         if op.isfile(op.join(self.image_dir, im)) and im.endswith(".jpg")])