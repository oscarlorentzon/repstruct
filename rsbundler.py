import os.path as op
import numpy as np
import sys
import getopt

from retrieval.flickrwrapper import FlickrWrapper
from analysis import pca, kclosest, process
from display import plothelper
from features.featuremode import FeatureMode
from features import sift, extract
from runmode import RunMode
from dataset.dataset import DataSet


class FlickrRsBundler:

    def __init__(self, api_key, tag):

        self.__data = DataSet(op.dirname(op.abspath(__file__)), tag)
        self.__flickr_wrapper = FlickrWrapper(api_key)

        self.__Y = None
        self.__closest30 = None
        self.__closest5 = None

    def run(self):
        self.download()
        self.extract()
        self.process()
        self.plot_image_pca()
        self.plot_result()
        
    def download(self):
        self.__flickr_wrapper.download(self.__data.image_dir, self.__data.tag, self.__data.processes)

    def extract(self):
        sift.extract(self.__data.images(), self.__data.image_dir, self.__data.feature_dir,
                     self.__data.processes)
        extract.extract(self.__data.images(), self.__data.image_dir, self.__data.feature_dir,
                        self.__data.descriptor_dir, self.__data.processes)

    def process(self):
        neutral_factor = self.__data.neutral_factor
        descriptor_weight = self.__data.descriptor_weight

        descriptors, descriptor_colors, random_colors = \
            extract.load_descriptors(self.__data.descriptor_dir, self.__data.images())
        
        if self.__data.feature_mode == FeatureMode.Colors:
            self.__Y = process.process(random_colors, neutral_factor)
        elif self.__data.feature_mode == FeatureMode.Descriptors:
            self.__Y = process.process(descriptors, neutral_factor)
        else:
            self.__Y = process.process_combined(descriptors, descriptor_colors, random_colors,
                                                descriptor_weight, neutral_factor)

        Y30 = self.__Y[:, :30]
        self.__closest30 = kclosest.k_closest(30, Y30)
        self.__closest5 = self.__closest30[kclosest.k_closest(5, Y30[self.__closest30, :])]
        
    def plot(self):
        plothelper.plot_images(self.__data.image_dir, self.__data.images()[self.__closest30], 3, 10)
        plothelper.plot_images(self.__data.image_dir, self.__data.images()[self.__closest5], 1, 5)

    def plot_pca(self):
        plothelper.plot_pca_projections(self.__Y, 1, 2)
        plothelper.plot_pca_projections(self.__Y, 3, 4)
        
    def plot_image_pca(self):
        plothelper.plot_pca_images(self.__data.image_dir, self.__data.images(), self.__Y, 1, 2)
        plothelper.plot_pca_images(self.__data.image_dir, self.__data.images(), self.__Y, 3, 4)

    def plot_result(self):
        plothelper.plot_result(self.__data.image_dir, self.__data.images(), self.__closest30, self.__closest5)
 
             
def main(argv):
    api_key = None
    tag = None
    run_mode = RunMode.Download
    
    helptext = """To run the bundler from command line enter:
              
               rsbundler.py -t <tag> -a <apikey>
    
               The following options are available:
    
               Required:
               -t : The tag to search for
               -a : The flickr api key
               
               Optional:
               -r : The run mode. Possible values:
                    d - Downloads, extracts, saves and processes images.
                        This is the default.
                    e - Extracts, saves and processes images.
                    o - Loads saved data and processes images.
               """
    
    try:
        opts, args = getopt.getopt(argv,"ha:t:f:r:", ["apikey=", "tag=", "featuremode=", "runmode="])
    except getopt.GetoptError:
        print helptext
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print helptext
            sys.exit()
        elif opt in ("-a", "--apikey"):
            api_key = arg
        elif opt in ("-t", "--tag"):
            tag = arg
        elif opt in ("-r", "--runmode"):
            if (arg == "e"):
                run_mode = RunMode.Extract
            elif (arg == "o"):
                run_mode = RunMode.Load
    
    if tag is None:   
        sys.exit("""Tag is required. Usage: """ + helptext)
    
    if api_key is None:
        with open("flickr_key.txt", "r") as f_out:
            api_key = f_out.readline().rstrip()

    bundler = FlickrRsBundler(api_key, tag)
    
    if run_mode == RunMode.Download:
        bundler.download()
        bundler.extract()
    elif run_mode == RunMode.Extract:
        bundler.extract()
        
    bundler.process()
    bundler.plot_image_pca()
    bundler.plot_result()


if __name__ == "__main__":
    main(sys.argv[1:])