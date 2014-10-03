import numpy as np

def neutral_sub_pca(X, neut_factor = 0.8):
    
    x_shape = X.shape
    row_count = x_shape[0]
    vector_length = x_shape[1]
    
    # Subracting a neutral vector for row in X before performing SVD
    N = np.sqrt(1.0/vector_length)*np.array([np.ones(vector_length),]*row_count)
    X_neut = X-neut_factor*N
    
    U,S,V = np.linalg.svd(X_neut)
    
    # projecting feature vectors on principal components
    y = np.dot(X_neut, V)
    
    return y, V