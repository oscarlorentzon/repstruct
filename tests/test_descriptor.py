import unittest
import numpy as np

from repstruct.features.descriptor import normalize_by_division, classify_euclidean, normalize, classify_cosine


class TestDescriptor(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass
    
    def testNormalize(self):
        v = [1, 1]
        X = np.array([v])
        
        result = normalize(X)
        
        norm = np.sqrt(np.sum(np.multiply(result, result), axis=1))
        
        self.assertLess(abs(1.0 - norm), 0.0000001, 'The norm is not one for the normalized array.')
        
    def testNormalizeMultipleVectors(self):
        v = [1, 1]
        X = np.array([v, v, v])
        
        result = normalize(X)
        
        norm = np.sqrt(np.sum(np.multiply(result, result), axis=1))
        
        self.assertLess(abs(1.0 - norm[0]), 0.0000001, 'The norm is not one for the normalized array.')
        self.assertLess(abs(1.0 - norm[1]), 0.0000001, 'The norm is not one for the normalized array.')
        self.assertLess(abs(1.0 - norm[2]), 0.0000001, 'The norm is not one for the normalized array.')

    def testNormalizeByDivision(self):
        l = [1, 2]
        v = np.array(l)
        n = np.array(l)
        
        result = normalize_by_division(v, n)
        
        self.assertLess(abs(1.0 - np.linalg.norm(result)), 0.0000001, 'The norm is not one for the normalized array.')
        self.assertEquals(result[0], result[1], 'The vector items should be equal after normalization.')  
        
    def testClassifyEuclideanOneVector(self):
        X = normalize(np.array([[1, 1]]))
        C = normalize(np.array([[1, 1], [0, 1]]))
        
        result = classify_euclidean(X, C)
        
        self.assertEqual(2, result.shape[0])
        self.assertEqual(1, result[0])
        self.assertEqual(0, result[1])
        
    def testClassifyEuclideanMultipleVectors(self):
        X = normalize(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        C = normalize(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        
        result = classify_euclidean(X, C)
        
        self.assertEqual(3, result.shape[0])
        self.assertEqual(3, np.sum(result))
        self.assertEqual(1, result[0])
        self.assertEqual(1, result[1])
        self.assertEqual(1, result[2])
        
    def testClassifyEuclideanMultipleVectorsSameCenter(self):
        X = normalize(np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]]))
        C = normalize(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        
        result = classify_euclidean(X, C)
        
        self.assertEqual(3, result.shape[0])
        self.assertEqual(3, np.sum(result))
        self.assertEqual(3, result[0])
        self.assertEqual(0, result[1])
        self.assertEqual(0, result[2])
        
    def testClassifyCosineOneVector(self):
        X = normalize(np.array([[1, 1]]))
        C = normalize(np.array([[1, 1], [0, 1]]))
        
        result = classify_cosine(X, C)
        
        self.assertEqual(2, result.shape[0])
        self.assertEqual(1, result[0])
        self.assertEqual(0, result[1])
        
    def testClassifyCosineMultipleVectors(self):
        X = normalize(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        C = normalize(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        
        result = classify_cosine(X, C)
        
        self.assertEqual(3, result.shape[0])
        self.assertEqual(3, np.sum(result))
        self.assertEqual(1, result[0])
        self.assertEqual(1, result[1])
        self.assertEqual(1, result[2])
        
    def testClassifyCosineMultipleVectorsSameCenter(self):
        X = normalize(np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]]))
        C = normalize(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        
        result = classify_cosine(X, C)
        
        self.assertEqual(3, result.shape[0])
        self.assertEqual(3, np.sum(result))
        self.assertEqual(3, result[0])
        self.assertEqual(0, result[1])
        self.assertEqual(0, result[2])


if __name__ == '__main__':
    unittest.main()