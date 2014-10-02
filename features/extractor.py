import scipy.io
from PIL import Image
import numpy as np

import sift as sift
import descriptor as descriptor

class Extractor:
    
    def __init__(self, image_files):
        self.image_files = image_files
    
    def extract(self):
        descriptor_data = scipy.io.loadmat('data/kmeans_descriptor_results.mat')
        
        descriptor_cluster_centers = descriptor_data.get('cbest')
        descriptor_cluster_center_indexes = descriptor_data.get('idxbest')
        
        # Create empty histogram array
        H = np.array([]).reshape(0, 1000)
        
        # extract descriptors for all images
        for image_file in self.image_files:
            
            image = Image.open(image_file)        
            shape = np.array(image).shape
            
            if len(shape) == 3:
                image = image.convert('L')
                
            locations, descriptors = sift.extract_feature_vectors(image)
            
            descriptor_norm = np.sqrt(np.sum(np.multiply(descriptors, descriptors)))
            
            descriptors = descriptors / descriptor_norm
            
            descriptor_histogram = descriptor.classify_descriptors(descriptors, descriptor_cluster_centers)
            
            print "Number of descriptors", (sum(descriptor_histogram))
            
            descriptor_histogram_norm, bins = np.histogram(descriptor_cluster_center_indexes, range(1,1002))
            
            normalized_descriptor_histogram = descriptor.normalize_descriptor_histogram(descriptor_histogram, descriptor_histogram_norm)
            
            H = np.vstack((H, normalized_descriptor_histogram))
        
        return H