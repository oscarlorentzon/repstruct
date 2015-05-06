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
        metric : The metric for which to perform the distance measure.
            Possible values are the ones defined for scipy.spatial.distance.pdist.
    
        Returns
        -------
        An array of row indices for the k closest row vectors.
    """
    
    d = pdist(V, metric)
    
    D = squareform(d)
    N = D.shape[0]
    
    row_dists = np.zeros(N)
    neighbors = np.zeros((N, k), dtype=np.int)

    # For each element, calculate the sum of distances 
    # to itself (=0) and its k-1 nearest neighbors.
    for i in range(0, N):
        row = D[i,:]
        
        # Get indices for the k closest items.
        indices = np.argsort(row)[:k]
        
        # Save the distance of the k closest items and the indices
        row_dists[i] = sum(row[indices])
        neighbors[i,:] = indices
        
    # Pick the k elements with the smallest distance sum.
    neighbors = neighbors[np.argsort(row_dists)[:k],:]
    
    internal_dists = np.zeros(k)
    
    # For each of the k elements, calculate the sum of
    # all internal distances in its neighborhood and 
    # return the neighborhood with the smallest sum.
    for i in range(0, k):
        D_indices = neighbors[i,:]
        internal_dists[i] = sum(sum(D[D_indices[:, None], D_indices]))

    # Return the row of neighbor indices with the smallest internal distance.
    return neighbors[np.argmin(internal_dists),:]