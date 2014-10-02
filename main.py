import os.path
from os import listdir
from os.path import isfile, join

import numpy as np
import scipy.io

from PIL import Image

from retrieval.flickrwrapper import FlickrWrapper
from display import plothelper

from sift import extract
from descriptor import classify_descriptors, normalize_descriptor_histogram
from analysis import kclosest, pca

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

a = np.array([]).reshape(0, 1000)

image_files = [join(image_dir,f) for f in listdir(image_dir) if isfile(join(image_dir,f))]

if extract_features:
    # get a list of all images    
    image_files = [join(image_dir,f) for f in listdir(image_dir) if isfile(join(image_dir,f))]
    
    descriptor_data = scipy.io.loadmat('data/kmeans_descriptor_results.mat')
    
    # extract descriptors for all images
    for image_file in image_files:
        
        image = Image.open(image_file)        
        shape = np.array(image).shape
        
        if len(shape) == 3:
            image = image.convert('L')
            
        locations, descriptors = extract.extract_feature_vectors(image)
        
        descriptor_norm = np.sqrt(np.sum(np.multiply(descriptors, descriptors)))
        
        descriptors = descriptors / descriptor_norm
        
        descriptor_histogram = classify_descriptors(descriptors, descriptor_data.get('cbest'))
        
        print "Number of descriptors", (sum(descriptor_histogram))
        
        descriptor_histogram_norm, bins = np.histogram(descriptor_data.get('idxbest'), range(1,1002))
        
        normalized_descriptor_histogram = normalize_descriptor_histogram(descriptor_histogram, descriptor_histogram_norm)
        
        a = np.vstack((a, normalized_descriptor_histogram))
    
    np.savetxt("deschists_" + tag + ".txt", a)
    
    print "descriptor histograms created"
    
if extract_features == False:
    a = np.loadtxt("deschists_" + tag + ".txt", float)

y, V = pca.generate_feature_vectors(a)

y30 = y[:,:30]
closest30 = kclosest.k_closest(30, y30)
ixs = kclosest.k_closest(5, y30[closest30,:])
closest5 = closest30[ixs]

plothelper.plot_images(np.array(image_files)[closest30], 3, 10)
plothelper.plot_images(np.array(image_files)[closest5], 1, 5)

plothelper.plot_pca_projections(y, 1, 2)
plothelper.plot_pca_projections(y, 3, 4)


