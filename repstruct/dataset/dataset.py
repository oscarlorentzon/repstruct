import os.path as op
import numpy as np
from os import makedirs, listdir

from configuration import *


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

        self.__feature = FeatureDataSet(self.__data_path, root_path)
        self.__descriptor = DescriptorDataSet(self.__data_path)
        self.__pca = PcaDataSet(self.__data_path, root_path)

        self.__image_path = self.__data_path + 'images/'
        self.__result_path = self.__data_path + 'results/'
        self.__plot_path = self.__data_path + 'plots/'

        # Create tag directories if not existing..
        for p in [self.__image_path, self.__result_path, self.__plot_path]:
            if not op.exists(p):
                makedirs(p)

    @property
    def tag(self):
        """ The image collection tag. """
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
    def feature(self):
        """ Feature data set. """
        return self.__feature

    @feature.setter
    def feature(self, feature):
        self.__feature = feature

    @property
    def descriptor(self):
        """ Descriptor data set. """
        return self.__descriptor

    @descriptor.setter
    def descriptor(self, descriptor):
        self.__descriptor = descriptor

    @property
    def pca(self):
        """ PCA data set. """
        return self.__pca

    @pca.setter
    def pca(self, pca):
        self.__pca = pca

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


class DataSetBase(object):

    def __init__(self, path, config):
        self._path = path
        self._config = config

        if not op.exists(self._path):
            makedirs(self._path)

    @property
    def path(self):
        """ The path to the data set. """
        return self._path

    @path.setter
    def path(self, path):
        self._path = path

    @property
    def config(self):
        """ Configuration. """
        return self._config


class FeatureDataSet(DataSetBase):

    def __init__(self, data_path, config_path):
        """ Initializes a feature data set.

        :param data_path: Path to data folder.
        :param config_path: Path to configuration file.
        """

        super(FeatureDataSet, self).__init__(op.join(data_path, 'features'), FeatureConfiguration(config_path))

    def save(self, image, locations, descriptors):
        """ Saves features for an image to .npz.

        :param image: Image name.
        :param locations: Descriptor locations.
        :param descriptors: Descriptor vectors.
        """

        np.savez(op.join(self._path, image + '.sift.npz'), locations=locations, descriptors=descriptors)

    def load(self, image):
        """ Loads features for an image from .npz.

        :param image: Image name.

        :return Feature locations and feature descriptors.
        """

        f = np.load(op.join(self._path, image + '.sift.npz'))

        return f['locations'], f['descriptors']


class DescriptorDataSet(DataSetBase):

    def __init__(self, data_path):
        """ Initializes a descriptor data set.

        :param data_path: Path to data folder.
        """

        super(DescriptorDataSet, self).__init__(op.join(data_path, 'descriptors'), {})

    def save(self, image, descriptors, descriptor_colors, random_colors):
        """ Saves bag of visual word descriptors to .npz.

        :param image: Image name.
        :param descriptors: Descriptor histogram.
        :param descriptor_colors: Histogram for colors in descriptor locations.
        :param random_colors: Histogram for colors in random locations.
        """

        np.savez(op.join(self._path, image + '.descriptors.npz'),
                 descriptors=descriptors,
                 descriptor_colors=descriptor_colors,
                 random_colors=random_colors)

    def load(self, image):
        """ Loads bag of visual word descriptors from .npz.

        :param image: Image name.

        :return descriptors: Descriptor histogram.
        :return descriptor_colors: Histogram for colors in descriptor locations.
        :return random_colors: Histogram for colors in random locations.
        """

        d = np.load(op.join(self._path, image + '.descriptors.npz'))

        return d['descriptors'], d['descriptor_colors'], d['random_colors']


class PcaDataSet(DataSetBase):

    def __init__(self, data_path, config_path):
        """ Initializes a pca data set.

        :param data_path: Path to data folder.
        :param config_path: Path to configuration file.
        """

        super(PcaDataSet, self).__init__(op.join(data_path, 'results'), PcaConfiguration(config_path))

    def save(self, images, pc_projections, principal_components):
        """ Saves result to .npz.

        :param images: Image names.
        :param pc_projections: The principal component projection arrays.
        :param principal_components: The principal components.
        """

        np.savez(op.join(self._path, 'principal_components.npz'),
                 images=images,
                 pc_projections=pc_projections,
                 principal_components=principal_components)

    def load(self):
        """ Loads principal components from .npz.

        :return images: Image names.
        :return pc_projections: The principal component projection arrays.
        :return principal_components: The principal components.
        """

        p = np.load(op.join(self._path, 'principal_components.npz'))

        return p['images'], p['pc_projections'], p['principal_components']