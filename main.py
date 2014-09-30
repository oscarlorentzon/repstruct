import urllib
import os.path
import numpy as np
from os import listdir
from os.path import isfile, join
from flickrdownloader import get_image_urls
from sift import extract
from PIL import Image
import scipy.io
from descriptor import classify_descriptors, normalize_descriptor_histogram
import feat_vec
from matplotlib.pyplot import plot, figure, show, axhline, axvline
from matplotlib.figure import Figure

download = False
extract_features = False
load = True

with open ("flickr_key.txt", "r") as myfile:
    api_key=myfile.read()

tag = 'goldengatebridge'

image_dir = os.path.dirname(os.path.abspath(__file__)) + "/images/" + tag

if download:
    # Get image urls from flickr api
    image_urls = get_image_urls(api_key, tag)
    
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # download images from flickr
    i = 1
    for image_url in image_urls:
        
        image_path = image_dir + "/" + tag + str(i) + ".jpg"
        urllib.urlretrieve(image_url, image_path)
        i += 1
    
    print "images downloaded"

a = np.array([]).reshape(0, 1000)

if extract_features:
    # get a list of all downloaded images    
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
    
    np.savetxt("deschists.txt", a)
    
    print "descriptor histograms created"
    
if load:
    a = np.loadtxt("deschists.txt", float)

y, V = feat_vec.generate_feature_vectors(a)

figure()
plot(y[:,1], y[:,2], '*')
axhline(0)
axvline(0)
show()

for value in y[:,2]:
    print value
