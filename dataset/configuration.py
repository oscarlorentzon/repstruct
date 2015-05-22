import yaml
import os.path as op

from features.featuremode import FeatureMode

class Configuration:

    def __init__(self, path=None, name='config.yaml'):
        """ Creates a configuration class by loading a configuration file.

        :param path: Path to config file.
        :param name: Name of the config file.
        :return:
        """

        if path is None:
            return

        # Load the configuration file and stores values in properties.
        with open(op.join(path, name)) as fin:
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

        self.__edge_threshold = self.__config['edge_threshold']
        self.__peak_threshold = self.__config['peak_threshold']

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
