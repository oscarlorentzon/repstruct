import numpy as np
from scipy.spatial.distance import pdist, squareform

def k_closest(k, V, metric='cosine'):
    """ Performs an approximate solution to the problem of
        finding the closest group of k elements in a set of 
        vectors.
        
        Parameters
        ----------
        k : The number of elements in the result set.
        V : A 2-D array with vectors in rows for which to find the 
            closest set of k vectors.
    
        Returns
        -------
        An array of row indices for the k closest row vectors.
    """
    
    d = pdist(V, metric)
    
    D = squareform(d)
    N = D.shape[0]
    
    total_dist = np.zeros(N)
    neighbours = np.zeros((N, k), dtype=np.int)

    # For each element, calculate the sum of distances 
    # to itself (=0) and its k-1 nearest neighbors.
    for i in range(0, N):
        row = D[i,:]
        
        # Get indices for the k closest items.
        indices = np.argsort(row)[:k]
        total_dist[i] = sum(row[indices])
        neighbours[i,:] = indices
        
    # Pick the k elements with the smallest distance sum.
    neighbours = neighbours[np.argsort(total_dist)[:k],:]
    
    # For each of these k elements, calculate the sum of
    # all internal distances in its neighborhood and 
    # return the neighborhood with the smallest sum.
    distsum2 = np.zeros(k)
    
    for i in range(0, k):
        
        a1 = neighbours[i,:]
        
        a = D[a1[:, None], a1]
        b = sum(a)
        c = sum(b)
        
        distsum2[i] = c

    mini = np.argmin(distsum2)
    
    ixs = neighbours[mini,:] 
    
    return ixs