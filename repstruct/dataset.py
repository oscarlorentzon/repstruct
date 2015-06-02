import os

from configuration import *


class DataSet(object):

    def __init__(self, tag, root_path=None):
        """ Creates a data set which holds paths, image list and config values.

        :param tag: Flickr tag.
        :param root_path: Path to directory including config file. Path where tags directory will be created.
        """

        self.__tag = tag
        self.__data_path = root_path + '/tags/' + self.__tag + '/' if root_path is not None else None

        self.__collection = CollectionDataSet(self.__data_path, root_path)
        self.__feature = FeatureDataSet(self.__data_path, root_path)
        self.__descriptor = DescriptorDataSet(self.__data_path)
        self.__pca = PcaDataSet(self.__data_path, root_path)
        self.__analysis = AnalysisDataSet(self.__data_path, root_path)
        self.__plot = PlotDataSet(self.__data_path, root_path)

    @property
    def tag(self):
        """ The image collection tag. """
        return self.__tag

    @tag.setter
    def tag(self, value):
        self.__tag = value

    @property
    def collection(self):
        """ Collection data set. """
        return self.__collection

    @collection.setter
    def collection(self, value):
        self.__collection = value

    @property
    def feature(self):
        """ Feature data set. """
        return self.__feature

    @feature.setter
    def feature(self, value):
        self.__feature = value

    @property
    def descriptor(self):
        """ Descriptor data set. """
        return self.__descriptor

    @descriptor.setter
    def descriptor(self, value):
        self.__descriptor = value

    @property
    def pca(self):
        """ PCA data set. """
        return self.__pca

    @pca.setter
    def pca(self, value):
        self.__pca = value

    @property
    def analysis(self):
        """ Analysis data set. """
        return self.__analysis

    @analysis.setter
    def analysis(self, value):
        self.__analysis = value

    @property
    def plot(self):
        """ Plot data set. """
        return self.__plot

    @plot.setter
    def plot(self, plot):
        self.__plot = plot


class DataSetBase(object):

    def __init__(self, path, folder, config):
        self._config = config

        if path is not None:
            self._path = op.join(path, folder)

            if not op.exists(self._path):
                os.makedirs(self._path)
        else:
            self._path = None

    @property
    def path(self):
        """ The path to the data set. """
        return self._path

    @path.setter
    def path(self, value):
        self._path = value

    @property
    def config(self):
        """ Configuration. """
        return self._config

    @config.setter
    def config(self, value):
        self._config = value

    def _save(self, file_name, **kwargs):
        """ Saves the key word arguments to npz file.

        :param file_name: File name without extension.
        :param kwargs: Keyword arguments.
        """

        np.savez(op.join(self._path, file_name + '.npz'), **kwargs)

    def _load(self, file_name):
        """ Loads npz data from file.

        :param file_name: File name without extension.
        :return: Loaded data.
        """

        return np.load(op.join(self._path, file_name + '.npz'))


class CollectionDataSet(DataSetBase):

    def __init__(self, data_path, config_path):
        """ Initializes a feature data set.

        :param data_path: Path to data folder.
        :param config_path: Path to configuration file.
        """

        super(CollectionDataSet, self).__init__(data_path, 'images', CollectionConfiguration(config_path))

    def images(self):
        """ Lists all images in the image directory.

        :return: List of image names.
        """

        return np.array([im for im in os.listdir(self._path)
                         if op.isfile(op.join(self._path, im)) and im.endswith(".jpg")])


class FeatureDataSet(DataSetBase):

    def __init__(self, data_path, config_path):
        """ Initializes a feature data set.

        :param data_path: Path to data folder.
        :param config_path: Path to configuration file.
        """

        super(FeatureDataSet, self).__init__(data_path, 'features', FeatureConfiguration(config_path))

    def save(self, image, locations, descriptors):
        """ Saves features for an image to file.

        :param image: Image name.
        :param locations: Feature locations.
        :param descriptors: Feature descriptor vectors.
        """

        self._save(image + '.sift', locations=locations, descriptors=descriptors)

    def load(self, image):
        """ Loads features for an image from file.

        :param image: Image name.

        :return Feature locations and feature descriptors.
        """

        f = self._load(image + '.sift')

        return f['locations'], f['descriptors']


class DescriptorDataSet(DataSetBase):

    def __init__(self, data_path):
        """ Initializes a descriptor data set.

        :param data_path: Path to data folder.
        """

        super(DescriptorDataSet, self).__init__(data_path, 'descriptors', {})

    def save(self, image, descriptors, descriptor_colors, random_colors):
        """ Saves bag of visual word descriptors to file.

        :param image: Image name.
        :param descriptors: Descriptor histogram.
        :param descriptor_colors: Histogram for colors in descriptor locations.
        :param random_colors: Histogram for colors in random locations.
        """

        self._save(image + '.descriptors',
                   descriptors=descriptors,
                   descriptor_colors=descriptor_colors,
                   random_colors=random_colors)

    def load(self, image):
        """ Loads bag of visual word descriptors from file.

        :param image: Image name.

        :return descriptors: Descriptor histogram.
        :return descriptor_colors: Histogram for colors in descriptor locations.
        :return random_colors: Histogram for colors in random locations.
        """

        d = self._load(image + '.descriptors')

        return d['descriptors'], d['descriptor_colors'], d['random_colors']


class PcaDataSet(DataSetBase):

    def __init__(self, data_path, config_path):
        """ Initializes a pca data set.

        :param data_path: Path to data folder.
        :param config_path: Path to configuration file.
        """

        super(PcaDataSet, self).__init__(data_path, 'results', PcaConfiguration(config_path))

    def save(self, images, pc_projections, principal_components):
        """ Saves result to file.

        :param images: Image names.
        :param pc_projections: The principal component projection arrays.
        :param principal_components: The principal components.
        """

        self._save('principal_components',
                   images=images,
                   pc_projections=pc_projections,
                   principal_components=principal_components)

    def load(self):
        """ Loads principal components from file.

        :return images: Image names.
        :return pc_projections: The principal component projection arrays.
        :return principal_components: The principal components.
        """

        p = self._load('principal_components')

        return p['images'], p['pc_projections'], p['principal_components']


class AnalysisDataSet(DataSetBase):

    def __init__(self, data_path, config_path):
        """ Initializes an analysis data set.

        :param data_path: Path to data folder.
        :param config_path: Path to configuration file.
        """

        super(AnalysisDataSet, self).__init__(data_path, 'results', AnalysisConfiguration(config_path))

    def save_closest(self, closest_group, representative):
        """ Saves closest group and representative indices result to file.

        :param closest_group: The image indices of the closest group.
        :param representative: The image indices of the representative group.
        """

        self._save('closest',
                   closest_group=closest_group,
                   representative=representative)

    def load_closest(self):
        """ Loads result from file.

        :param file_path: The result folder.

        :return closest_group: The image indices of the closest group.
        :return representative: The image indices of the representative group.
        """

        c = self._load('closest')

        return c['closest_group'], c['representative']

    def save_structures(self, centroids, structures):
        """ Saves k-means structure results to file.

        :param centroids: The cluster centroids.
        :param structures: Array of structures containing indices for images connected to each cluster centroid.
        """

        self._save('structures',
                   centroids=centroids,
                   structures=structures)

    def load_structures(self):
        """ Loads result from file.

        :return centroids: The cluster centroids.
        :return structures: Array of structures containing indices for images connected to each cluster centroid.
        """

        s = self._load('structures')

        return s['centroids'], s['structures']

    def save_scored_structures(self, scored_structures):
        """ Saves scored structures to file.

        :param scored_structures: Arrays of structures ordered base on score.
        """

        self._save('scored_structures', scored_structures=scored_structures)

    def load_scored_structures(self):
        """ Loads scored structures from file.

        :return scored_structures: Array of structures ordered base on score.
        """

        s = self._load('scored_structures')

        return s['scored_structures']


class PlotDataSet(DataSetBase):

    def __init__(self, data_path, config_path):
        """ Initializes a plot data set.

        :param data_path: Path to data folder.
        :param config_path: Path to configuration file.
        """

        super(PlotDataSet, self).__init__(data_path, 'plots', PlotConfiguration(config_path))