import scipy.io
from PIL import Image
import numpy as np

from enum import Enum

import matplotlib.colors as mc

import sift as sift
import features.descriptor as desc

class FeatureMode(Enum):
    All = 0
    Descriptors = 1
    Colors = 2
    

def extract(image_files, mode=FeatureMode.All):
    descriptor_data = scipy.io.loadmat('data/kmeans_descriptor_results.mat')
    color_data = scipy.io.loadmat('data/kmeans_color_results.mat')
    
    D = None
    C_rand = None
    C_desc = None
    
    if mode != FeatureMode.Colors:
        # Get descriptor training data cluster centers and descriptor training data 
        # cluster center indexes and create histogram.
        desc_cc = descriptor_data.get('cbest')
        desc_cc_norm = np.histogram(descriptor_data.get('idxbest'), range(1, desc_cc.shape[0] + 2))[0]
        
        # Create empty histogram array
        D = np.array([]).reshape(0, desc_cc.shape[0])

    if mode != FeatureMode.Descriptors:
        color_cc = color_data.get('ccbest')
        color_cc_norm = np.histogram(color_data.get('idxcbest'), range(1, color_cc.shape[0] + 2))[0]
        
        # Retrieve Gaussian distributed random points for color retrieval.
        gaussians = scipy.io.loadmat('data/gaussians.mat').get('rands')
        x = np.mod((1+gaussians['x'][0,0]/2.3263)/2, 1)[:, 0]
        y = np.mod((1+gaussians['y'][0,0]/2.3263)/2, 1)[:, 0]
    
        C_rand = np.array([]).reshape(0, color_cc.shape[0])
        
    if mode == FeatureMode.All:
        C_desc = np.array([]).reshape(0, color_cc.shape[0])
    
    # extract descriptors and colors for all images
    for image_file in image_files:
        
        image = Image.open(image_file)        
        shape = np.array(image).shape
        
        if mode != FeatureMode.Colors:
            if len(shape) == 3:
                gray_image = image.convert('L')
                
            locs, descs = sift.extract_feature_vectors(gray_image)
            
            descs = desc.normalize(descs)
            desc_cc = desc.normalize(desc_cc)
            desc_hist = desc.classify_cosine(descs, desc_cc)
            
            norm_desc_hist = desc.normalize_by_division(desc_hist, desc_cc_norm)
            
            D = np.vstack((D, norm_desc_hist))
            
            print "Number of descs", descs.shape[0]
            
            if mode == FeatureMode.All:
                
                if len(shape) == 3:
                    im = np.array(image)/255.0
                
                    colors_desc = get_rgb_from_locs(locs[:, 1], locs[:, 0], im)
                    colors_desc_coords = rgb_to_hs_coords(colors_desc)
                    colors_desc_hist = desc.classify_euclidean(colors_desc_coords, color_cc)
                    colors_desc_hist_norm = desc.normalize_by_division(colors_desc_hist, color_cc_norm)
                else:
                    colors_desc_hist_norm = create_NaN_array(1, color_cc.shape[0])
                    
                C_desc = np.vstack((C_desc, colors_desc_hist_norm))
        
        if mode != FeatureMode.Descriptors:
            if len(shape) == 3:
                im = np.array(image)/255.0
                colors_rand = get_rgb_from_locs(im.shape[0]*np.array(y), im.shape[1]*np.array(x), im)
                colors_rand_coords = rgb_to_hs_coords(colors_rand)
                colors_rand_hist = desc.classify_euclidean(colors_rand_coords, color_cc)
                colors_rand_hist_norm = desc.normalize_by_division(colors_rand_hist, color_cc_norm)
            else:
                colors_rand_hist_norm = create_NaN_array(1, color_cc.shape[0])
        
            C_rand = np.vstack((C_rand, colors_rand_hist_norm))
      
    if C_rand is not None:
        C_norm = np.linalg.norm(C_rand, axis=1)
        
        C_real = np.mean(C_rand[~np.isnan(C_norm), :], axis=0)
        C_real = C_real / np.linalg.norm(C_real)
        
        C_rand[np.isnan(C_norm), :] = np.tile(C_real, (sum(np.isnan(C_norm)), 1))
      
    if C_desc is not None:
        C_norm = np.linalg.norm(C_desc, axis=1)
        
        C_real = np.mean(C_desc[~np.isnan(C_norm), :], axis=0)
        C_real = C_real / np.linalg.norm(C_real)
        
        C_desc[np.isnan(C_norm), :] = np.tile(C_real, (sum(np.isnan(C_norm)), 1))

    if mode == FeatureMode.All:
        w = 0.725
        cw = (1-w)/2   
        H = np.hstack((np.sqrt(w)*D, np.hstack((np.sqrt(cw)*C_desc, np.sqrt(cw)*C_rand))))
        
        N = create_neutral_vector(np.array([[desc_cc.shape[0], np.sqrt(w)],[color_cc.shape[0], np.sqrt(cw)],[color_cc.shape[0], np.sqrt(cw)]]), H.shape[0])
    
    if mode == FeatureMode.Colors:
        H = C_rand
        N = create_neutral_vector(np.array([[color_cc.shape[0], 1]]), H.shape[0])
        
    if mode == FeatureMode.Descriptors:
        H = D
        N = create_neutral_vector(np.array([[desc_cc.shape[0], 1]]), H.shape[0])
            
    return H, N

def create_neutral_vector(D, rows):
    
    A = np.array([]).reshape(rows, 0)
    
    for d in D:
        A = np.concatenate((A, d[1]*np.sqrt(1.0/d[0])*np.array([np.ones(d[0]),]*rows)), axis=1)
    
    return A

def create_NaN_array(rows, cols):
    nan_array = np.empty((rows, cols))
    nan_array[:] = np.NAN
    return nan_array

def rgb_to_hs_coords(rgb):
    hsv = mc.rgb_to_hsv(rgb)
    
    x = np.multiply(hsv[:,1], np.cos(2*np.pi*hsv[:,0]))
    y = np.multiply(hsv[:,1], np.sin(2*np.pi*hsv[:,0]))
    
    return np.vstack((x, y)).transpose()

def get_rgb_from_locs(locs_r, locs_c, im):
    locs_row = np.empty((locs_r.shape[0],), dtype='int')
    locs_col = np.empty((locs_r.shape[0],), dtype='int')
    np.floor(locs_r, out=locs_row)
    np.floor(locs_c, out=locs_col)
    return im[locs_row, locs_col]









