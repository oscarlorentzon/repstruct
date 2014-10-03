import scipy.io
from PIL import Image
import numpy as np

import sift as sift
import features.descriptor as desc

class Extractor:
    
    def __init__(self, image_files):
        self.image_files = image_files
    
    def extract(self):
        descriptor_data = scipy.io.loadmat('data/kmeans_descriptor_results.mat')
        
        # Descriptor training data cluster centers
        desc_cc = descriptor_data.get('cbest')

        # Gets the training data descriptor cluster center indexes and create histogram
        desc_cc_norm, bins = np.histogram(descriptor_data.get('idxbest'), range(1, 1002))
        
        # Create empty histogram array
        H = np.array([]).reshape(0, 1000)
        
        # extract descs for all images
        for image_file in self.image_files:
            
            image = Image.open(image_file)        
            shape = np.array(image).shape
            
            if len(shape) == 3:
                image = image.convert('L')
                
            locs, descs = sift.extract_feature_vectors(image)
            
            descs = desc.normalize(descs)
            desc_cc = desc.normalize(desc_cc)
            desc_hist = desc.classify(descs, desc_cc)
            
            norm_desc_hist = desc.normalize_by_division(desc_hist, desc_cc_norm)

            H = np.vstack((H, norm_desc_hist))
            
            print "Number of descs", descs.shape[0] #(sum(desc_hist))
            
        return H