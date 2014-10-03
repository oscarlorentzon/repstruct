import numpy as np

def normalize(X):
    """ Normalizes the rows of a 2-D array.
            
        Parameters
        ----------
        X : A 2-D array.
    
        Returns
        -------
        A 2-D array with the rows normalized.
    """
    
    norm = np.linalg.norm(X, axis=1)
    
    return np.divide(X, np.tile(np.array([norm]).transpose(), [1, X.shape[1]]))

def classify(X, C):
    """ Classifies the row vectors of X on the cluster center
        row vectors of C using cosine similarity.
        
        Parameters
        ----------
        X : A 2-D array.
        C : A 2-D array of cluster centers.
    
        Returns
        -------
        A histogram of indices for the cluster centers of the classified row vectors.
        
        Notes
        -----
        The length of the row vectors in X must be the same as the length of the cluster
        center row vectors i C.
    
    """
    
    # Clusters row vectors into classes with centers on the unit sphere using cosine similarity
    # by getting the center which has the highest dot product value with each vector row.
    indices = np.argmax(np.dot(X, C.transpose()), 1)
    
    cc_count = C.shape[0]
    hist, bins = np.histogram(indices, range(1, cc_count + 2))
    
    return hist


def normalize_by_division(v, n):
    """ Divides each element of the vector v with the corresponding 
        element of the normalization vector n.
         
        Parameters
        ----------
        v : A 1-D array.
        n : A 1-D array of normalization values.
    
        Returns
        -------
        An array with elements normalized by the normalization vector n which
        is L2 normalized.
        
        Notes
        -----
        The length of the vectors v and n must be equal.
    
    """
    
    divided = np.divide([float(i) for i in v], n)
    
    return divided / np.linalg.norm(divided)
    



