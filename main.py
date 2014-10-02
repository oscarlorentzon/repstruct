import os.path
from os import listdir
from os.path import isfile, join
import numpy as np

from features.extractor import Extractor
from retrieval.flickrwrapper import FlickrWrapper
from display import plothelper
from analysis import pca, kclosest

# Steps to run
download = False
extract_features = False

# Parameters
with open ("flickr_key.txt", "r") as myfile:
    api_key=myfile.readline().rstrip()
tag = 'steppyramid'
image_dir = os.path.dirname(os.path.abspath(__file__)) + "/images/" + tag

if download:
    flickrWrapper = FlickrWrapper(api_key)
    flickrWrapper.download_images(image_dir, tag)

image_files = [join(image_dir,f) for f in listdir(image_dir) if isfile(join(image_dir,f))]

if extract_features:
    
    extractor = Extractor(image_files)
    H = extractor.extract()
    
    #np.savetxt("deschists_" + tag + ".txt", a)
    
    print "descriptor histograms created"
else:
    H = np.loadtxt("deschists_" + tag + ".txt", float)

y, V = pca.generate_feature_vectors(H)

y30 = y[:,:30]
closest30 = kclosest.k_closest(30, y30)
ixs = kclosest.k_closest(5, y30[closest30,:])
closest5 = closest30[ixs]

plothelper.plot_images(np.array(image_files)[closest30], 3, 10)
plothelper.plot_images(np.array(image_files)[closest5], 1, 5)

#plothelper.plot_pca_projections(y, 1, 2)
#plothelper.plot_pca_projections(y, 3, 4)


