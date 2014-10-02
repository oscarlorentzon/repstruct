import numpy as np
from scipy.spatial.distance import pdist, squareform

def k_closest(k, V, metric='cosine'):
    """ Returns an approximate solution to the problem of
        finding the closest group of k elements in a set
        where the internal distances are given by Dv.
    """
    
    d = pdist(V, metric)
    
    D = squareform(d)
    N = D.shape[0]
    
    distsum = np.zeros(N)
    distsum2 = np.zeros(k)
    ix = np.zeros((N, k), dtype=np.int)

    # For each element, calculate the sum of distances 
    # to itself (=0) and its k-1 nearest neighbours
    for i in range(0, N):
        indices = np.argsort(D[i,:])
        
        vect = D[indices,i]
        distsum[i] = sum(vect[:k])
        
        ix[i,:] = indices[:k]
        
    # Pick the k elements with the smallest distance sum
    indices2 = np.argsort(distsum)
    ix = ix[indices2[:k],:]
    
    # For each of these k elements, calculate the sum of
    # all internal distances in its neighbourhood and 
    # return the neighbourhood with the smallest sum
    for i in range(0, k):
        
        a1 = ix[i,:]
        
        a = D[a1[:, None], a1]
        b = sum(a)
        c = sum(b)
        
        distsum2[i] = c

    mini = np.argmin(distsum2)
    
    ixs = ix[mini,:] 
    
    return ixs