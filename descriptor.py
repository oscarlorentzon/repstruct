import numpy as np

def get_descriptor_histogram(descriptors, cluster_center_locations):
    
    
    cluster_center_norm =  np.sqrt(np.sum(np.multiply(cluster_center_locations, cluster_center_locations), 1))
    
    normalized_cluster_center_locations = np.divide(cluster_center_locations, np.tile(cluster_center_norm, [128,1]).transpose())

    indices = get_closest_cluster_center_indices(descriptors, normalized_cluster_center_locations)
    
    # TODO: 1000 is length of cluster center locations
    descriptor_histogram = np.histogram(indices, range(1,1001))
    
    return descriptor_histogram


def get_closest_cluster_center_indices(descriptors, normalized_cluster_center_locations):
    
    return np.argmax(np.dot(descriptors, normalized_cluster_center_locations.transpose()), 1)


