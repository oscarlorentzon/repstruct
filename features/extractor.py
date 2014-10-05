import scipy.io
from PIL import Image
import numpy as np

import sift as sift
import features.descriptor as desc
import display.plothelper as ph

def extract(image_files):
    descriptor_data = scipy.io.loadmat('data/kmeans_descriptor_results.mat')
    color_data = scipy.io.loadmat('data/kmeans_color_results.mat')
    
    # Descriptor training data cluster centers
    desc_cc = descriptor_data.get('cbest')
    
    # Gets the training data descriptor cluster center indexes and create histogram
    desc_cc_norm, b1 = np.histogram(descriptor_data.get('idxbest'), range(1, desc_cc.shape[0] + 2))
    
    color_cc = color_data.get('ccbest')
    
    color_cc_norm, b2 = np.histogram(color_data.get('idxcbest'), range(1, color_cc.shape[0] + 2))
    
    # Retrieve Gaussian distributed random points for color retrieval.
    gaussians = scipy.io.loadmat('data/gaussians.mat').get('rands')
    x = np.mod((1+gaussians['x'][0,0]/2.3263)/2, 1)
    y = np.mod((1+gaussians['y'][0,0]/2.3263)/2, 1)
    
    # Create empty histogram array
    H = np.array([]).reshape(0, desc_cc.shape[0] + 2*color_cc.shape[0])
    
    # extract descriptors and colors for all images
    for image_file in image_files:
        
        image = Image.open(image_file)        
        shape = np.array(image).shape
        
        if len(shape) == 3:
            image = image.convert('L')
            
        locs, descs = sift.extract_feature_vectors(image)
        
        descs = desc.normalize(descs)
        desc_cc = desc.normalize(desc_cc)
        desc_hist = desc.classify(descs, desc_cc)
        
        norm_desc_hist = desc.normalize_by_division(desc_hist, desc_cc_norm)
        
        
        
        #% Return color in interest points
        #if any(abs(colorimage(:,1)-colorimage(:,3)) > 0.001)
        #    [m,n] = size(image);
        #    load rands
        #    rx = m*mod((1+rands.x/2.3263)/2,1);
        #    ry = n*mod((1+rands.y/2.3263)/2,1);
        #    colors = colorimage((round(locs(:,2))-1)*m + round(locs(:,1)),:);
        #    rcolors = colorimage(floor(ry)*m + ceil(rx),:);
        #    
        #end

        H = np.vstack((H, norm_desc_hist))
        
        print "Number of descs", descs.shape[0] #(sum(desc_hist))
        
    return H










