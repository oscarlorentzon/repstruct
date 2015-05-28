import yaml
import os.path as op
import numpy as np

from repstruct.featuremode import FeatureMode


class Configuration:

    def __init__(self, path=None, name='config.yaml'):
        """ Creates a configuration class by loading a configuration file.

        :param path: Path to config file.
        :param name: Name of the config file.
        """

        if path is not None:
            # Load the configuration file and stores values in properties.
            with open(op.join(path, name)) as fin:
                self.__config = yaml.load(fin)
        else:
            self.__config = {}

        # Properties for SIFT extraction.
        self.__edge_threshold = self.__config.get('edge_threshold', 10)
        self.__peak_threshold = self.__config.get('peak_threshold', 0.01)

        # Properties for principal component analysis.
        self.__descriptor_weight = self.__config.get('descriptor_weight', 0.725)
        self.__neutral_factor = self.__config.get('neutral_factor', 0.8)

        feature_mode = self.__config.get('feature_mode', 'ALL').upper()
        if feature_mode == 'ALL':
            self.__feature_mode = FeatureMode.All
        elif feature_mode == 'DESCRIPTORS':
            self.__feature_mode = FeatureMode.Descriptors
        elif feature_mode == 'COLORS':
            self.__feature_mode = FeatureMode.Colors
        else:
            raise ValueError('Unknown feature mode (must be ALL, DESCRIPTORS or COLORS)')

        # Properties for feature vector distance measuring.
        self.__pc_projection_count = self.__config.get('pc_projection_count', 30)
        self.__closest_group = self.__config.get('closest_group', 0.3)
        self.__representative = self.__config.get('representative', 0.05)

        # Properties for determining all structures using k-means.
        self.__clusters = self.__config.get('clusters', 8)
        self.__runs = self.__config.get('runs', 500)
        self.__iterations = self.__config.get('iterations', 100)

        # General properties.
        self.__processes = self.__config.get('processes', 8)
        self.__collection_count = self.__config.get('collection_count', 100)

        # Plot properties
        self.__save_plot = self.__config.get('save_plot', False)
        self.__image_dimension = self.__config.get('image_dimension', 100)
        self.__columns = self.__config.get('columns', 10)
        self.__ticks = self.__config.get('ticks', False)
        self.__pc_plots = np.array(self.__config.get('pc_plots', [[2, 3], [4, 5]])) - 1

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
    def edge_threshold(self):
        """ SIFT edge threshold. """
        return self.__edge_threshold

    @edge_threshold.setter
    def edge_threshold(self, edge_threshold):
        self.__edge_threshold = edge_threshold

    @property
    def peak_threshold(self):
        """ SIFT peak threshold. """
        return self.__peak_threshold

    @peak_threshold.setter
    def peak_threshold(self, peak_threshold):
        self.__peak_threshold = peak_threshold

    @property
    def pc_projection_count(self):
        """ Number of principal component projections used for distance measuring. """
        return self.__pc_projection_count

    @pc_projection_count.setter
    def pc_projection_count(self, pc_projection_count):
        self.__pc_projection_count = pc_projection_count

    @property
    def closest_group(self):
        """ Proportion of the images to be included in the closest group. """
        return self.__closest_group

    @closest_group.setter
    def closest_group(self, closest_group):
        self.__closest_group = closest_group

    @property
    def representative(self):
        """ Proportion of the images to be determined as representative. """
        return self.__representative

    @representative.setter
    def representative(self, representative):
        self.__representative = representative

    @property
    def clusters(self):
        """ Number of cluster centroids. """
        return self.__clusters

    @clusters.setter
    def clusters(self, clusters):
        self.__clusters = clusters

    @property
    def runs(self):
        """ Number of times to run the k-means algorithm. The run with lowest distortion is chosen. """
        return self.__runs

    @runs.setter
    def runs(self, runs):
        self.__runs = runs

    @property
    def iterations(self):
        """ Number of iterations for each run of k-means. """
        return self.__iterations

    @iterations.setter
    def iterations(self, iterations):
        self.__iterations = iterations

    @property
    def processes(self):
        """ Number of parallel processes for downloading and extraction. """
        return self.__processes

    @processes.setter
    def processes(self, processes):
        self.__processes = processes

    @property
    def collection_count(self):
        """ Size of collection. Number of images to be downloaded. """
        return self.__collection_count

    @collection_count.setter
    def collection_count(self, collection_count):
        self.__collection_count = collection_count

    @property
    def save_plot(self):
        """ Boolean specifying if plots should be saved to file. """
        return self.__save_plot

    @save_plot.setter
    def save_plot(self, save_plot):
        self.__save_plot = save_plot

    @property
    def image_dimension(self):
        """ Dimension of the longest side of images in plots. """
        return self.__image_dimension

    @image_dimension.setter
    def image_dimension(self, image_dimension):
        self.__image_dimension = image_dimension

    @property
    def columns(self):
        """ Number of columns in result plot. """
        return self.__columns

    @columns.setter
    def columns(self, columns):
        self.__columns = columns

    @property
    def ticks(self):
        """ Boolean specifying if the principal component projection plots should have ticks. """
        return self.__ticks

    @ticks.setter
    def ticks(self, ticks):
        self.__ticks = ticks

    @property
    def pc_plots(self):
        """ Principal components for which to plot the projections against. """
        return self.__pc_plots

    @pc_plots.setter
    def pc_plots(self, pc_plots):
        self.__pc_plots = pc_plots