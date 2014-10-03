import numpy as np

def neutral_sub_pca(X, neut_factor=0.8):
    """ Performs PCA by singular value decomposition after subtracting
        a neutral vector with a specified factor. 
        
        Parameters
        ----------
        X : A 2-D array with normalized row vectors.
        neut_factor : The factor of the neutralization vector.
      
        Returns
        -------
        Y : A 2-D array of projections of the row vectors of X on the 
            principal components.
        V : The principal components of X.
    """
    
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