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
    H = np.array([]).reshape(0, desc_cc.shape[0])
    C = np.array([]).reshape(0, 2*color_cc.shape[0])
    
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
        
        H = np.vstack((H, norm_desc_hist))
        
        print "Number of descs", descs.shape[0]
        
        im = np.array(image)
        
        if len(shape) == 3:
            locs_x = np.empty((locs.shape[0],), dtype='int')
            locs_y = np.empty((locs.shape[0],), dtype='int')
            np.floor(locs[:, 1], out=locs_x)
            np.floor(locs[:, 0], out=locs_y)
            colors_desc = im[locs_x, locs_y]/255.0
            
            rand_x = np.empty((x.shape[0],), dtype='int')
            rand_y = np.empty((y.shape[0],), dtype='int')
            im_shape = im.shape
            np.floor(im_shape[0]*np.array(x), out=rand_x)
            np.floor(im_shape[1]*np.array(y), out=rand_y)
            
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
        else:
            aanormhist = np.empty((1, color_cc.shape[0]))
            aanormhist[:] = np.NAN
            bbnormhist = np.empty((1, color_cc.shape[0]))
            bbnormhist[:] = np.NAN
        
        C = np.vstack((C, np.hstack((aanormhist,bbnormhist))))
        
    
    nnn = np.linalg.norm(C, axis=1)
    
    uuu = ~np.isnan(nnn)
    
    CCC = C[uuu, :]
    
    mmm = np.mean(CCC, axis=0)
    
    mmm = np.sqrt(2) * mmm / np.linalg.norm(mmm)
    
    iii = np.isnan(nnn)
    
    C[iii, :] = np.tile(mmm, (sum(np.isnan(nnn)), 1))

    w = 0.725
    cw = (1-w)/2   
    vvv = np.hstack((np.sqrt(w)*H, np.sqrt(cw)*C))
    
    oooo = np.linalg.norm(C, axis=1)
        
    N = create_neutral_vector(np.array([[desc_cc.shape[0], np.sqrt(w)],[color_cc.shape[0], np.sqrt(cw)],[color_cc.shape[0], np.sqrt(cw)]]), H.shape[0])
            
    return vvv, N

def create_neutral_vector(D, rows):
    
    A = np.array([]).reshape(rows, 0)
    
    for d in D:
        A = np.concatenate((A, d[1]*np.sqrt(1.0/d[0])*np.array([np.ones(d[0]),]*rows)), axis=1)
    
    return A










