import unittest
import numpy as np
from features.descriptor import normalize_by_division, normalize


class DescirptorTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def normalizeByDivision(self):
        l = [1, 2]
        v = np.array(l)
        n = np.array(l)
        
        result = normalize_by_division(v, n)
        
        self.assertLess(abs(1.0 - np.linalg.norm(result)), 0.0000001, 'The norm is not one for the normalized array.')
        self.assertEquals(result[0], result[1], 'The vector items should be equal after normalization.')  


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()