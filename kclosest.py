import numpy as np
from scipy.spatial.distance import pdist, squareform

def test(k, V, metric='cosine'):
    
    d = pdist(V, metric)
    
    D = squareform(d)
    N = D.shape[0]
    
    distsum = np.zeros(N)
    distsum2 = np.zeros(k)
    ix = np.zeros((N, k), dtype=np.int)

    for i in range(0, N):
        indices = np.argsort(D[i,:])
        
        vect = D[indices,i]
        distsum[i] = sum(vect[:k])
        
        ix[i,:] = indices[:k]
        
    
    indices2 = np.argsort(distsum)
    ix = ix[indices2[:k],:]
    
    for i in range(0, k):
        
        a1 = ix[i,:]
        
        a = D[a1[:, None], a1]
        b = sum(a)
        c = sum(b)
        
        distsum2[i] = c

    mini = np.argmin(distsum2)
    
    ixs = ix[mini,:] 
    
    return ixs
    
    
    #% Returns an approximate solution to the problem of
#% finding the closest group of k elements in a set
#% where the internal distances are given by Dv.
#%
#% Oscar Lorentzon and Nils Lundahl 2009
#%--------------------------------------------------------------------------
#
#    D = squareform(Dv);
#    N = size(D,1);
#    distsum = zeros(N,1);
#    distsum2 = zeros(k,1);
#    ix = zeros(N,k);
#
#    % For each element, calculate the sum of distances 
#    % to itself (=0) and its k-1 nearest neighbours
#    for i = 1:N
#        [mindists,ixf] = sort(D(i,:));
#        distsum(i) = sum(mindists(1:k));
#        ix(i,:) = ixf(1:k);
#    end
#
#    % Pick the k elements with the smallest distance sum
#    [distsum,I] = sort(distsum);
#    ix = ix(I(1:k),:);
#
#    % For each of these k elements, calculate the sum of
#    % all internal distances in its neighbourhood and 
#    % return the neighbourhood with the smallest sum
#    for i = 1:k
#        distsum2(i) = sum(sum(D(ix(i,:),ix(i,:))));
#    end
#
#    [duh,I] = min(distsum2);
#    ixs = ix(I,:);
#end