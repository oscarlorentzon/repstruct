import os.path as op
from os import listdir
import numpy as np

from retrieval.flickrwrapper import FlickrWrapper
from features import extractor
from analysis import pca, kclosest
from display import plothelper
from features.extractor import FeatureMode

class FlickrRsBundler:
    
    hist_file = "{0}_deschists.txt"
    neut_file = "{0}_neut.txt"
    
    def __init__(self, api_key, tag, neut_factor=0.8):
        self.flickrWrapper = FlickrWrapper(api_key)
        self.tag = tag
        self.image_dir = op.dirname(op.abspath(__file__)) + "/images/" + self.tag + "/"
        self.neut_factor = neut_factor
        
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
        self.H, self.N = extractor.extract(self.image_files, FeatureMode.Descriptors)
        
    def save(self):
        np.savetxt(self.image_dir + self.hist_file.format(self.tag), self.H)
        np.savetxt(self.image_dir + self.neut_file.format(self.tag), self.N)
        
    def load(self):
        self.files()
        self.H = np.loadtxt(self.image_dir + self.hist_file.format(self.tag), float)
        self.N = np.loadtxt(self.image_dir + self.neut_file.format(self.tag), float)
        
    def process(self):        
        #self.Y, V = pca.neutral_sub_pca(self.H)
        
        self.Y, V = pca.neutral_sub_pca_vector(self.H, self.neut_factor*self.N)

        Y30 = self.Y[:,:30]
        self.closest30 = kclosest.k_closest(30, Y30)
        self.closest5 = self.closest30[kclosest.k_closest(5, Y30[self.closest30,:])]
        
    def plot(self):
        plothelper.plot_images(np.array(self.image_files)[self.closest30], 3, 10)
        plothelper.plot_images(np.array(self.image_files)[self.closest5], 1, 5)
        plothelper.plot_pca_projections(self.Y, 1, 2)
        plothelper.plot_pca_projections(self.Y, 3, 4)
        
        