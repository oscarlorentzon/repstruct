import os.path as op
from os import listdir, makedirs
import numpy as np
import sys, getopt

from retrieval.flickrwrapper import FlickrWrapper
from features.extract import extract, create_neutral_vector
from analysis import pca, kclosest
from display import plothelper
from features.featuremode import FeatureMode
from features import sift
from runmode import RunMode


class FlickrRsBundler:
    
    __desc_file = "desc.txt"
    __color_desc_file = "colordesc.txt"
    __color_rand_file = "colorrand.txt"
    
    def __init__(self, api_key, tag):
        self.__flickrWrapper = FlickrWrapper(api_key)
        self.__tag = tag

        self.__data_dir = op.dirname(op.abspath(__file__)) + "/tags/" + self.__tag + "/"
        self.__image_dir = self.__data_dir + "images/"
        self.__feature_dir = self.__data_dir + "features/"
        self.__descriptor_dir = self.__data_dir + "descriptors/"

        self.__image_files = None

        self.__D = None
        self.__C_desc = None
        self.__C_rand = None

        self.__Y = None
        self.__closest30 = None
        self.__closest5 = None

        if not op.exists(self.__image_dir):
            makedirs(self.__image_dir)

        if not op.exists(self.__feature_dir):
            makedirs(self.__feature_dir)

        if not op.exists(self.__descriptor_dir):
            makedirs(self.__descriptor_dir)

    def run(self):
        self.download()
        self.extract()
        self.save()
        self.process()
        self.plot()

    def __images(self):
        if self.__image_files is not None:
            return self.__image_files

        self.__image_files =\
            [im for im in listdir(self.__image_dir) if op.isfile(op.join(self.__image_dir,im)) and im.endswith(".jpg")]

        return self.__image_files
        
    def download(self):
        self.__flickrWrapper.download(self.__image_dir, self.__tag)

    def extract(self):
        sift.extract(self.__images(), self.__image_dir, self.__feature_dir)
        self.__D, self.__C_desc, self.__C_rand = extract(self.__images(), self.__image_dir, self.__feature_dir, self.__descriptor_dir)
        
    def save(self):
        np.savetxt(self.__descriptor_dir + self.__desc_file, self.__D)
        np.savetxt(self.__descriptor_dir + self.__color_desc_file, self.__C_desc)
        np.savetxt(self.__descriptor_dir + self.__color_rand_file, self.__C_rand)
        
    def load(self):
        self.__D = np.loadtxt(self.__descriptor_dir + self.__desc_file, float)
        self.__C_desc = np.loadtxt(self.__descriptor_dir + self.__color_desc_file, float)
        self.__C_rand = np.loadtxt(self.__descriptor_dir + self.__color_rand_file, float)
        
    def process(self, mode=FeatureMode.All, neut_factor=0.8, d_weight=0.725):
        
        if mode == FeatureMode.Colors:
            N = create_neutral_vector(np.array([[self.__C_rand.shape[1], 1]]), self.__C_rand.shape[0])
            F = self.__C_rand
        elif mode == FeatureMode.Descriptors:
            N = create_neutral_vector(np.array([[self.__D.shape[1], 1]]), self.__D.shape[0])
            F = self.__D
        else:
            c_weight = (1-d_weight)/2  
            N = create_neutral_vector(
                np.array([[self.__D.shape[1], np.sqrt(d_weight)],
                          [self.__C_desc.shape[1], np.sqrt(c_weight)],
                          [self.__C_rand.shape[1], np.sqrt(c_weight)]]),
                self.__D.shape[0])
            F = np.hstack((np.sqrt(d_weight)*self.__D,
                           np.hstack((np.sqrt(c_weight)*self.__C_desc, np.sqrt(c_weight)*self.__C_rand))))
        
        self.__Y, V = pca.neutral_sub_pca_vector(F, neut_factor*N)

        Y30 = self.__Y[:, :30]
        self.__closest30 = kclosest.k_closest(30, Y30)
        self.__closest5 = self.__closest30[kclosest.k_closest(5, Y30[self.__closest30, :])]
        
    def plot(self):
        plothelper.plot_images(self.__image_dir, np.array(self.__images())[self.__closest30], 3, 10)
        plothelper.plot_images(self.__image_dir, np.array(self.__images())[self.__closest5], 1, 5)

    def plot_pca(self):
        plothelper.plot_pca_projections(self.__Y, 1, 2)
        plothelper.plot_pca_projections(self.__Y, 3, 4)
        
    def plot_image_pca(self):
        plothelper.plot_pca_images(self.__image_dir, self.__images(), self.__Y, 1, 2)
        plothelper.plot_pca_images(self.__image_dir, self.__images(), self.__Y, 3, 4)

    def plot_result(self):
        plothelper.plot_result(self.__images(), self.__closest30, self.__closest5, self.__image_dir)
 
             
def main(argv):
    api_key = None
    tag = None
    feature_mode = FeatureMode.All
    run_mode = RunMode.Download
    
    helptext = """To run the bundler from command line enter:
              
               rsbundler.py -t <tag> -a <apikey>
    
               The following options are available:
    
               Required:
               -t : The tag to search for
               -a : The flickr api key
               
               Optional:
               -f : The feature mode. Possible values:
                    a - Both descriptors and colors. This is the default.
                    d - Descriptors only
                    c - Colors only
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
        elif opt in ("-f", "--featuremode"):
            if (arg == "d"):
                feature_mode = FeatureMode.Descriptors
            elif (arg == "c"):
                feature_mode = FeatureMode.Colors
        elif opt in ("-r", "--runmode"):
            if (arg == "e"):
                run_mode = RunMode.Extract
            elif (arg == "o"):
                run_mode = RunMode.Load
    
    if tag is None:   
        sys.exit("""Tag is required. Usage: """ + helptext)
    
    if api_key is None:
        with open ("flickr_key.txt", "r") as myfile: api_key=myfile.readline().rstrip()

    bundler = FlickrRsBundler(api_key, tag)
    
    if (run_mode == RunMode.Download):
        bundler.download()
        bundler.extract()
        bundler.save()
    elif (run_mode == RunMode.Extract):
        bundler.extract()
        bundler.save()
    else:
        bundler.load()
        
    bundler.process(feature_mode)
    bundler.plot_image_pca()
    bundler.plot_result()


if __name__ == "__main__":
    main(sys.argv[1:])