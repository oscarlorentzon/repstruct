import numpy as np
from numpy import argmin

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

def classify_cosine(X, C):
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
    return np.histogram(indices, range(1, cc_count + 2))[0]

def classify_euclidean(X, C):
    #% Clusters colours into colour classes using the L2 metric
    #% size(X) = m x dim
    #% size(cc) = n x dim
    #% size(d) = m x n x dim

    m = X.shape[0]
    dim = X.shape[1]
    n = C.shape[0]
    
    # 1263 x 200 x 2
    X3 = np.zeros((m,n,dim))
    c3 = np.zeros((m,n,dim))
    
    X3 = X.reshape(m, 1, dim).repeat(n, axis=1)
    
    c3 = C.reshape(1, n, dim).repeat(m, axis=0)
    
    d = np.sum(np.power(X3-c3, 2), axis=2)

    a = np.argmin(d, axis=1)  
    
    hist = np.histogram(a, range(1, n + 2))[0] 
    
    return hist   
    #    [m,dim] = size(X);
    #    n = size(cc,1);
    #    X3 = zeros(m,n,dim);
    #    c3 = X3;
    #    X3(:,1,:) = X;
    #    X3 = X3(:,ones(1,n),:);
    #    c3(1,:,:) = cc;
    #    c3 = c3(ones(1,m),:,:);
    #
    #    d = sum((X3-c3).^2,3);
    #
    #    [meh,idX] = min(d,[],2);
    
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
    



