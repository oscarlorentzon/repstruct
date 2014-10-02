import numpy as np

def generate_feature_vectors(X):
    
    U,S,V = np.linalg.svd(X)
    
    # projecting feature vectors on principal components
    y = np.dot(X, V)
    
    return y, V