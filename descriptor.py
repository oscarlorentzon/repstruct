from numpy import multiply, sqrt, sum


def test(descriptors, cluster_center_locations, cluster_center_indeces):
    
    a = multiply(cluster_center_locations, cluster_center_locations)
    
    cluster_center_norm =  sqrt(sum(multiply(cluster_center_locations, cluster_center_locations), 1))
    
    a = 1





#% Clustering descriptors and creating a histogramvector for each image
#    cnorm = sqrt(sum(cbest.*cbest,2));
#    cbest = cbest./cnorm(:,ones(128,1));
#    k = length(descriptors);
#    a = size(cbest,1);
#    xd = zeros(k,a);
#    hixindic = zeros(1,k);
#    histnormering = hist(idxbest,1:a);
#    for i = 1:k
#        indexes = klustra(descriptors{i},cbest);
#        hists = hist(indexes,1:a);
#        % Eliminate images with less than 100 descriptors
#        if sum(hists) >= 100
#            xd(i,:) = hists;
#            hixindic(i) = 1;
#        end
#    end
#    histindexes = find(hixindic);
#    xd = xd(histindexes,:);
    


#function idX = klustra(X,c)
#% Clusters descriptors into classes with centroids 
#% on the unit sphere using cosine similarity
#    [duh,idX] = max(X*c',[],2);
#end
