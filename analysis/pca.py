import numpy as np

def neutral_sub_pca(X, neut_factor=0.8):
    
    X_shape = X.shape
    row_count = X_shape[0]
    vector_length = X_shape[1]
    
    # Subtracting a neutral vector for each row in X before performing SVD.
    N = np.sqrt(1.0/vector_length)*np.array([np.ones(vector_length),]*row_count)
    X_neut = X-neut_factor*N
    
    U,S,V = np.linalg.svd(X_neut)
    
    # Projecting feature vectors on principal components.
    Y = np.dot(X_neut, V)
    
    return Y, V