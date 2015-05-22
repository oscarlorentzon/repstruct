import yaml
import os.path as op

from features.featuremode import FeatureMode

class Configuration:

    def __init__(self, path=None, name='config.yaml'):
        """ Creates a configuration class by loading a configuration file.

        :param path: Path to config file.
        :param name: Name of the config file.
        """

        if path is None:
            return

        # Load the configuration file and stores values in properties.
        with open(op.join(path, name)) as fin:
            self.__config = yaml.load(fin)

        # Properties for SIFT extraction.
        self.__edge_threshold = self.__config['edge_threshold']
        self.__peak_threshold = self.__config['peak_threshold']

        # Properties for principal component analysis.
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

        # Properties for feature vector distance measuring.
        self.__pc_projection_count = self.__config['pc_projection_count']
        self.__closest_group = self.__config['closest_group']
        self.__representative = self.__config['representative']

        # General properties.
        self.__processes = self.__config['processes']

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
