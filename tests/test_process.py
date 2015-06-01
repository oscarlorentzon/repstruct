import unittest
import math

from repstruct.analysis.process import *
from repstruct.dataset import *


class TestProcess(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testCreateNeutralVector(self):
        weight = 0.3
        ds = np.array([[1, weight], [1, 1 - weight]])
        rows = 1

        result = create_neutral_vector(ds, rows)

        self.assertEqual(rows, result.shape[0])
        self.assertEqual(np.sum(ds[:, 0]), result.shape[1])

        norm = np.linalg.norm(result)

        self.assertLess(abs(1.0 - norm), 0.0000001, 'The norm is not one for the normalized array.')

    def testCreateNeutralVectorMultiple(self):
        weight1 = 0.3
        weight2 = 0.2
        weight3 = 0.1
        weight4 = 0.4
        ds = np.array([[4, weight1],
                       [3, weight2],
                       [2, weight3],
                       [1, weight4]])
        rows = 3

        result = create_neutral_vector(ds, rows)

        self.assertEqual(rows, result.shape[0])
        self.assertEqual(np.sum(ds[:, 0]), result.shape[1])

        norm = np.linalg.norm(result, axis=1)
        max_norm_error = np.max(np.abs(norm - 1.0))

        self.assertLess(max_norm_error, 0.0000001, 'The norm is not one for the normalized array.')

    def testCreateNeutralVectorRaises(self):
        ds = np.array([[1, 0.5]])
        self.assertRaises(AssertionError, create_neutral_vector, ds, 1)

    def testSetNanRowsToNormalizedMean(self):
        x = np.array([[1., 1.], np.empty(2)])
        x[1, :] = np.NaN

        result = set_nan_rows_to_normalized_mean(x)

        norm = np.linalg.norm(result[1, :])

        self.assertLess(abs(1.0 - norm), 0.0000001, 'The norm is not one for the normalized array.')
        self.assertLess(abs(result[1, 0] - result[1, 1]), 0.0000001)

    def testSetNanRowsToNormalizedMeanMultiple(self):
        x = np.array([[1., 0.], [0., 1.], np.empty(2)])
        x[2, :] = np.NaN

        result = set_nan_rows_to_normalized_mean(x)

        norm = np.linalg.norm(result[2, :])

        self.assertLess(abs(1.0 - norm), 0.0000001, 'The norm is not one for the normalized array.')
        self.assertLess(abs(result[2, 0] - result[2, 1]), 0.0000001)


if __name__ == '__main__':
    unittest.main()
