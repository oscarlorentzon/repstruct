from numpy import *

def generate_feature_vectors(X):
    
    U,S,V = linalg.svd(X)
    
    # projecting feature vectors on principal components
    y = dot(X, V)
    
    return y, V
        