import yaml
import os.path as op
from os import makedirs, listdir


class DataSet:

    def __init__(self, root_path, tag):

        self.__data_dir = root_path + "/tags/" + tag + "/"

        self.image_dir = self.__data_dir + "images/"
        self.feature_dir = self.__data_dir + "features/"
        self.descriptor_dir = self.__data_dir + "descriptors/"

        self.tag = tag

        config_file = op.join(root_path, 'config.yaml')
        with open(config_file) as fin:
            self.config = yaml.load(fin)

        for p in [self.image_dir, self.feature_dir, self.descriptor_dir]:
            if not op.exists(p):
                makedirs(p)

    def images(self):
        return [im for im in listdir(self.image_dir)
                if op.isfile(op.join(self.image_dir, im)) and im.endswith(".jpg")]