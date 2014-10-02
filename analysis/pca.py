import numpy as np

def generate_feature_vectors(X):
    
    x_shape = X.shape
    vector_length = x_shape[1]
    nbr_of_vectors = x_shape[0]
    norm_neutral_vector = 0.8
    neutral_vector = np.sqrt(1.0 / vector_length) * np.ones(vector_length)
    
    neutral_2d_array = np.array([neutral_vector,]*nbr_of_vectors)
    
    desc_subtracted_with_norm = X - norm_neutral_vector * neutral_2d_array
    
    U,S,V = np.linalg.svd(desc_subtracted_with_norm)
    
    # projecting feature vectors on principal components
    y = np.dot(desc_subtracted_with_norm, V)
    
    return y, V