import os.path as op
import numpy as np
from os import makedirs, listdir

from configuration import Configuration


class DataSet:

    def __init__(self, tag, root_path=None):
        """ Creates a data set which holds paths, image list and config values.

        :param tag: Flickr tag.
        :param root_path: Path to directory including config file. Path where tags directory will be created.
        """

        self.__tag = tag
        self.__config = Configuration(root_path)

        if root_path is None:
            return

        self.__data_path = root_path + '/tags/' + self.__tag + '/'

        self.__image_path = self.__data_path + 'images/'
        self.__feature_path = self.__data_path + 'features/'
        self.__descriptor_path = self.__data_path + 'descriptors/'
        self.__result_path = self.__data_path + 'results/'
        self.__plot_path = self.__data_path + 'plots/'

        # Create tag directories if not existing..
        for p in [self.__image_path, self.__feature_path, self.__descriptor_path, self.__result_path, self.__plot_path]:
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
    def plot_path(self):
        """ The path to the plot directory. """
        return self.__plot_path

    @plot_path.setter
    def plot_path(self, plot_path):
        self.__plot_path = plot_path

    @property
    def config(self):
        """ Configuration. """
        return self.__config

    def images(self):
        """ Lists all images in the image directory.

        :return: List of image names.
        """

        return np.array([im for im in listdir(self.image_path)
                         if op.isfile(op.join(self.image_path, im)) and im.endswith(".jpg")])