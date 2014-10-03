import os.path as op
from os import listdir
import numpy as np

from retrieval.flickrwrapper import FlickrWrapper
from features import extractor
from analysis import pca, kclosest
from display import plothelper

class FlickrRsBundler:
    
    hist_file = "deschists_{0}.txt"
    
    def __init__(self, api_key, tag):
        self.flickrWrapper = FlickrWrapper(api_key)
        self.tag = tag
        self.image_dir = op.dirname(op.abspath(__file__)) + "/images/" + self.tag
        
    def run(self):
        self.download()
        self.extract()
        self.save()
        self.process()
        self.plot()
        
    def download(self):
        self.flickrWrapper.download(self.image_dir, self.tag)
        
    def files(self):
        self.image_files = [op.join(self.image_dir,f) for f in listdir(self.image_dir) if op.isfile(op.join(self.image_dir,f))]
    
    def extract(self):
        self.files()
        self.H = extractor.extract(self.image_files)
        
    def save(self):
        np.savetxt(self.hist_file.format(self.tag), self.H)
        
    def load(self):
        self.files()
        self.H = np.loadtxt(self.hist_file.format(self.tag), float)
        
    def process(self):
        self.Y, V = pca.neutral_sub_pca(self.H)

        Y30 = self.Y[:,:30]
        self.closest30 = kclosest.k_closest(30, Y30)
        self.closest5 = self.closest30[kclosest.k_closest(5, Y30[self.closest30,:])]
        
    def plot(self):
        plothelper.plot_images(np.array(self.image_files)[self.closest30], 3, 10)
        plothelper.plot_images(np.array(self.image_files)[self.closest5], 1, 5)
        plothelper.plot_pca_projections(self.Y, 1, 2)
        plothelper.plot_pca_projections(self.Y, 3, 4)
        
        