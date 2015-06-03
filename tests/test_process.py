import unittest

from mock import Mock

import numpy as np

import repstruct.analysis.process as process
import repstruct.dataset as dataset


class TestProcess(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testCreateNeutralVector(self):
        weight = 0.3
        ds = np.array([[1, weight], [1, 1 - weight]])
        rows = 1

        result = process.create_neutral_vector(ds, rows)

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

        result = process.create_neutral_vector(ds, rows)

        self.assertEqual(rows, result.shape[0])
        self.assertEqual(np.sum(ds[:, 0]), result.shape[1])

        norm = np.linalg.norm(result, axis=1)
        max_norm_error = np.max(np.abs(norm - 1.0))

        self.assertLess(max_norm_error, 0.0000001, 'The norm is not one for the normalized array.')

    def testCreateNeutralVectorRaises(self):
        ds = np.array([[1, 0.5]])
        self.assertRaises(AssertionError, process.create_neutral_vector, ds, 1)

    def testSetNanRowsToNormalizedMean(self):
        x = np.array([[1., 1.], np.empty(2)])
        x[1, :] = np.NaN

        result = process.set_nan_rows_to_normalized_mean(x)

        norm = np.linalg.norm(result[1, :])

        self.assertLess(abs(1.0 - norm), 0.0000001, 'The norm is not one for the normalized array.')
        self.assertLess(abs(result[1, 0] - result[1, 1]), 0.0000001)

    def testSetNanRowsToNormalizedMeanMultiple(self):
        x = np.array([[1., 0.], [0., 1.], np.empty(2)])
        x[2, :] = np.NaN

        result = process.set_nan_rows_to_normalized_mean(x)

        norm = np.linalg.norm(result[2, :])

        self.assertLess(abs(1.0 - norm), 0.0000001, 'The norm is not one for the normalized array.')
        self.assertLess(abs(result[2, 0] - result[2, 1]), 0.0000001)

    def testLoadDescriptors(self):
        images = np.array(['im1', 'im2', 'im3'])

        descriptor_data = dataset.DescriptorDataSet(None)
        descriptor_data.load = Mock(return_value=(np.array([0]), np.array([1]), np.array([2])))

        result = process.load_descriptors(descriptor_data, images)

        desc_res = result[0]
        desc_col_res = result[1]
        rand_col_res = result[2]

        self.assertEqual(len(images), desc_res.shape[0])
        self.assertEqual(len(images), desc_col_res.shape[0])
        self.assertEqual(len(images), rand_col_res.shape[0])

        self.assertTrue(np.all(desc_res == 0))
        self.assertTrue(np.all(desc_col_res == 1))
        self.assertTrue(np.all(rand_col_res== 2))


if __name__ == '__main__':
    unittest.main()
