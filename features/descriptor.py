import numpy as np

def classify_descriptors(descriptors, cluster_center_locations):
    """
    
    """
    
    cluster_center_norm =  np.sqrt(np.sum(np.multiply(cluster_center_locations, cluster_center_locations), 1))
    
    normalized_cluster_center_locations = np.divide(cluster_center_locations, np.tile(cluster_center_norm, [128,1]).transpose())

    indices = get_closest_cluster_center_indices(descriptors, normalized_cluster_center_locations)
    
    # TODO: 1000 is length of cluster center locations
    descriptor_histogram, bins = np.histogram(indices, range(1,1002))
    
    return descriptor_histogram


def get_closest_cluster_center_indices(descriptors, normalized_cluster_center_locations):
    """ Clusters descriptors into classes with centroids 
        on the unit sphere using cosine similarity 
    """
    
    return np.argmax(np.dot(descriptors, normalized_cluster_center_locations.transpose()), 1)

def normalize_descriptor_histogram(descriptor_histogram, descriptor_histogram_norm):
    """
  
    """
    
    normalized_descriptor_histogram = np.divide([float(i) for i in descriptor_histogram], descriptor_histogram_norm)
    
    descriptor_histogram_norm_scalar = np.sqrt(np.sum(np.multiply(normalized_descriptor_histogram, normalized_descriptor_histogram)))

    final_normalized_descriptor_histogram = normalized_descriptor_histogram / descriptor_histogram_norm_scalar
    
    norm_neutral_vector = 0.8
    
    # ones(1000)    
    neutral_vector = np.sqrt(1.0 / 1000) * np.array([1.0 for i in range(1, 1001)])
    
    desc_subtracted_with_norm = final_normalized_descriptor_histogram - norm_neutral_vector * neutral_vector
    
    return desc_subtracted_with_norm
    



