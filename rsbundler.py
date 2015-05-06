import os.path as op
from os import listdir
import numpy as np
import sys, getopt

from retrieval.flickrwrapper import FlickrWrapper
from features.extract import extract, create_neutral_vector
from analysis import pca, kclosest
from display import plothelper
from features.featuremode import FeatureMode
from runmode import RunMode


class FlickrRsBundler:
    
    desc_file = "{0}_desc.txt"
    color_desc_file = "{0}_colordesc.txt"
    color_rand_file = "{0}_colorrand.txt"
    
    def __init__(self, api_key, tag):
        self.flickrWrapper = FlickrWrapper(api_key)
        self.tag = tag
        self.data_dir = op.dirname(op.abspath(__file__)) + "/tags/" + self.tag + "/"
        self.image_dir = self.data_dir + "images/"

        self.image_files = None

        self.D = None
        self.C_desc = None
        self.C_rand = None

        self.Y = None
        self.closest30 = None
        self.closest5 = None

    def run(self):
        self.download()
        self.extract()
        self.save()
        self.process()
        self.plot()
        
    def download(self):
        self.flickrWrapper.download(self.image_dir, self.tag)
        
    def files(self):
        self.image_files = [op.join(self.image_dir,f) for f in listdir(self.image_dir) if op.isfile(op.join(self.image_dir,f)) and f.endswith(".jpg")]
    
    def extract(self):
        self.files()
        self.D, self.C_desc, self.C_rand = extract(self.image_files)
        
    def save(self):
        np.savetxt(self.image_dir + self.desc_file.format(self.tag), self.D)
        np.savetxt(self.image_dir + self.color_desc_file.format(self.tag), self.C_desc)
        np.savetxt(self.image_dir + self.color_rand_file.format(self.tag), self.C_rand)
        
    def load(self):
        self.files()
        self.D = np.loadtxt(self.image_dir + self.desc_file.format(self.tag), float)
        self.C_desc = np.loadtxt(self.image_dir + self.color_desc_file.format(self.tag), float)
        self.C_rand = np.loadtxt(self.image_dir + self.color_rand_file.format(self.tag), float)
        
    def process(self, mode=FeatureMode.All, neut_factor=0.8, d_weight=0.725):
        
        if mode == FeatureMode.Colors:
            N = create_neutral_vector(np.array([[self.C_rand.shape[1], 1]]), self.C_rand.shape[0])
            F = self.C_rand
        elif mode == FeatureMode.Descriptors:
            N = create_neutral_vector(np.array([[self.D.shape[1], 1]]), self.D.shape[0])
            F = self.D
        else:
            c_weight = (1-d_weight)/2  
            N = create_neutral_vector(
                np.array([[self.D.shape[1], np.sqrt(d_weight)],
                          [self.C_desc.shape[1], np.sqrt(c_weight)],
                          [self.C_rand.shape[1], np.sqrt(c_weight)]]),
                self.D.shape[0])
            F = np.hstack((np.sqrt(d_weight)*self.D,
                           np.hstack((np.sqrt(c_weight)*self.C_desc, np.sqrt(c_weight)*self.C_rand))))
        
        self.Y, V = pca.neutral_sub_pca_vector(F, neut_factor*N)

        Y30 = self.Y[:,:30]
        self.closest30 = kclosest.k_closest(30, Y30)
        self.closest5 = self.closest30[kclosest.k_closest(5, Y30[self.closest30,:])]
        
    def plot(self):
        plothelper.plot_images(np.array(self.image_files)[self.closest30], 3, 10)
        plothelper.plot_images(np.array(self.image_files)[self.closest5], 1, 5)
        
    def plot_pca(self):
        plothelper.plot_pca_projections(self.Y, 1, 2)
        plothelper.plot_pca_projections(self.Y, 3, 4)
 
             
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
    bundler.plot()
    bundler.plot_pca()


if __name__ == "__main__":
    main(sys.argv[1:])