import numpy as np

import process


def neutral_sub_pca(X, neut_factor=0.8):
    """ Performs PCA by singular value decomposition after
        subtracting a neutral vector with a specified factor.

    :param X: A 2-D array with normalized row vectors.
    :param neut_factor: The factor of the neutralization vector.

    :return Y: A 2-D array of projections of the row vectors
               of X on the principal components.
    :return V: The principal components of X.
    """
    
    X_shape = X.shape
    row_count = X_shape[0]
    vector_length = X_shape[1]
    
    # Subtracting a neutral vector for each row in X before performing SVD.
    N = process.create_neutral_vector(np.array([[vector_length, 1.]]), row_count)
    X_neut = X-neut_factor*N
    
    U, S, VT = np.linalg.svd(X_neut)

    # Projecting feature vectors on principal components.
    V = VT.T
    Y = np.dot(X_neut, V)
    
    return Y, V


def neutral_sub_pca_vector(X, N):
    """ Performs PCA by singular value decomposition after subtracting
        a neutral vector with a specified factor. 

    :param X: A 2-D array with normalized row vectors.
    :param neut_factor: The factor of the neutralization vector.

    :return Y: A 2-D array of projections of the row vectors of X on
               the principal components.
    :return V: The principal components of X.
    """
    
    # Subtracting a neutral vector for each row in X before performing SVD.
    X_neut = X-N
    
    U, S, VT = np.linalg.svd(X_neut)

    # Projecting feature vectors on principal components.
    V = VT.T
    Y = np.dot(X_neut, V)
    
    return Y, V