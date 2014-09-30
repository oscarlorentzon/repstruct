from numpy import loadtxt
import os

def extract_feature_vectors(image,params="--edge-thresh 10 --peak-thresh 1"):
    """ Process an image and save the results in a file. """
    
    tmp_dir = os.path.dirname(os.path.abspath(__file__)) + "/tmp/"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    
    tmp_pgm = tmp_dir + "tmp.pgm"
    tmp_key = tmp_dir + "tmp.sift"
    
    image.save(tmp_pgm)
    
    cmmd = str("sift " + tmp_pgm + " --output=" + tmp_key + " " + params)
    os.system(cmmd)

    # print "processed", "to", tmp_key
    
    result = loadtxt(tmp_key)
    return result[:,:4], result[:,4:]
    
    

