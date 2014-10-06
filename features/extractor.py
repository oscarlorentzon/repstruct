import scipy.io
from PIL import Image
import numpy as np

import matplotlib.colors as mc

import sift as sift
import features.descriptor as desc
import display.plothelper as ph

def extract(image_files):
    descriptor_data = scipy.io.loadmat('data/kmeans_descriptor_results.mat')
    color_data = scipy.io.loadmat('data/kmeans_color_results.mat')
    
    # Get descriptor training data cluster centers and descriptor training data 
    # cluster center indexes and create histogram.
    desc_cc = descriptor_data.get('cbest')
    desc_cc_norm = np.histogram(descriptor_data.get('idxbest'), range(1, desc_cc.shape[0] + 2))[0]
    
    
    color_cc = color_data.get('ccbest')
    color_cc_norm = np.histogram(color_data.get('idxcbest'), range(1, color_cc.shape[0] + 2))[0]
    
    # Retrieve Gaussian distributed random points for color retrieval.
    gaussians = scipy.io.loadmat('data/gaussians.mat').get('rands')
    x = np.mod((1+gaussians['x'][0,0]/2.3263)/2, 1)[:, 0]
    y = np.mod((1+gaussians['y'][0,0]/2.3263)/2, 1)[:, 0]
    
    # Create empty histogram array
    H = np.array([]).reshape(0, desc_cc.shape[0] + 2*color_cc.shape[0])
    
    # extract descriptors and colors for all images
    for image_file in image_files:
        
        image = Image.open(image_file)        
        shape = np.array(image).shape
        
        if len(shape) == 3:
            gray_image = image.convert('L')
            
        locs, descs = sift.extract_feature_vectors(gray_image)
        
        descs = desc.normalize(descs)
        desc_cc = desc.normalize(desc_cc)
        desc_hist = desc.classify_cosine(descs, desc_cc)
        
        norm_desc_hist = desc.normalize_by_division(desc_hist, desc_cc_norm)
        
        im = np.array(image)
        
        locs_x = np.empty((locs.shape[0],), dtype='int')
        locs_y = np.empty((locs.shape[0],), dtype='int')
        np.round(locs[:, 1], out=locs_x)
        np.round(locs[:, 0], out=locs_y)
        
        colors_desc = im[locs_x, locs_y, :]/255.0
        
        rand_x = np.empty((x.shape[0],), dtype='int')
        rand_y = np.empty((y.shape[0],), dtype='int')
        im_shape = im.shape
        np.round(im_shape[0]*np.array(x), out=rand_x)
        np.round(im_shape[1]*np.array(y), out=rand_y)
        
        colors_rand = im[rand_x, rand_y]/255.0
        
        a = mc.rgb_to_hsv(colors_desc)
        b = mc.rgb_to_hsv(colors_rand)
        
        acos = np.multiply(a[:,1], np.cos(2*np.pi*a[:,0]))
        asin = np.multiply(a[:,1], np.sin(2*np.pi*a[:,0]))
        
        bcos = np.multiply(b[:,1], np.cos(2*np.pi*b[:,0]))
        bsin = np.multiply(b[:,1], np.sin(2*np.pi*b[:,0]))
        
        #ph.plot_points(acos, asin)
        #ph.plot_points(bcos, bsin)
        
        #ph.plot_points(color_cc[:, 0], color_cc[:, 1])
        
        aa = np.vstack((acos, asin)).transpose()
        bb = np.vstack((bcos, bsin)).transpose()
        
        aahist = desc.classify_euclidean(aa, color_cc)
        bbhist = desc.classify_euclidean(bb, color_cc)
        
        aanormhist = desc.normalize_by_division(aahist, color_cc_norm)
        bbnormhist = desc.normalize_by_division(bbhist, color_cc_norm)

        w = 0.725
        cw = (1-w)/2
        
        vvv = np.hstack((np.sqrt(w)*norm_desc_hist, np.sqrt(cw)*aanormhist, np.sqrt(cw)*bbnormhist))
        
        vvva = np.linalg.norm(vvv)

        H = np.vstack((H, vvv))
        
        print "Number of descs", descs.shape[0]
        
    return H










