import unittest
import numpy as np

from mock import Mock, patch, PropertyMock

import repstruct.analysis.process as process
from repstruct.configuration import FeatureMode
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

    @patch('repstruct.analysis.pca.neutral_sub_pca_vector')
    @patch('repstruct.analysis.process.create_neutral_vector')
    def testProcessFeatures(self, neut_mock, pca_mock):
        neut_mock.return_value = np.array([1, 1])
        pca_mock.return_value = ('projs', 'pcs')

        features = np.array([[2, 2]])
        neut_factor = 2
        process.process_features(features, neut_factor)

        call_args = pca_mock.call_args[0]

        self.assertSequenceEqual(list(features[0]), list(call_args[0][0]))
        self.assertSequenceEqual(list(neut_factor * neut_mock.return_value), list(call_args[1]))

    @patch('repstruct.analysis.process.process_combined_features')
    @patch('repstruct.analysis.process.process_features')
    @patch('repstruct.analysis.process.load_descriptors')
    def testProcessColors(self, load_mock, proc_feat_mock, proc_comb_mock):
        data = dataset.DataSet('tag')
        data.collection = PropertyMock()
        data.collection.images = Mock(return_value=np.array(['im1', 'im2']))
        data.pca = PropertyMock()
        data.pca.config = PropertyMock()
        data.pca.config.feature_mode = FeatureMode.Colors
        data.pca.config.neutral_factor = 3
        data.pca.save = Mock()

        load_mock.return_value = ('desc', 'desc_color', 'rand_color')
        proc_feat_mock.return_value = ('projections', 'comp')
        proc_comb_mock.return_value = ('projections', 'comp')

        process.process(data)

        proc_feat_mock.assert_called_with('rand_color', data.pca.config.neutral_factor)
        self.assertEqual(0, proc_comb_mock.call_count)

    @patch('repstruct.analysis.process.process_combined_features')
    @patch('repstruct.analysis.process.process_features')
    @patch('repstruct.analysis.process.load_descriptors')
    def testProcessDescriptors(self, load_mock, proc_feat_mock, proc_comb_mock):
        data = dataset.DataSet('tag')
        data.collection = PropertyMock()
        data.collection.images = Mock(return_value=np.array(['im1', 'im2']))
        data.pca = PropertyMock()
        data.pca.config = PropertyMock()
        data.pca.config.feature_mode = FeatureMode.Descriptors
        data.pca.config.neutral_factor = 3
        data.pca.save = Mock()

        load_mock.return_value = ('desc', 'desc_color', 'rand_color')
        proc_feat_mock.return_value = ('projections', 'comp')
        proc_comb_mock.return_value = ('projections', 'comp')

        process.process(data)

        proc_feat_mock.assert_called_with('desc', data.pca.config.neutral_factor)
        self.assertEqual(0, proc_comb_mock.call_count)

    @patch('repstruct.analysis.process.process_combined_features')
    @patch('repstruct.analysis.process.process_features')
    @patch('repstruct.analysis.process.load_descriptors')
    def testProcessAll(self, load_mock, proc_feat_mock, proc_comb_mock):
        data = dataset.DataSet('tag')
        data.collection = PropertyMock()
        data.collection.images = Mock(return_value=np.array(['im1', 'im2']))
        data.pca = PropertyMock()
        data.pca.config = PropertyMock()
        data.pca.config.feature_mode = FeatureMode.All
        data.pca.config.neutral_factor = 5
        data.pca.config.descriptor_weight = 8
        data.pca.save = Mock()

        load_mock.return_value = ('desc', 'desc_color', 'rand_color')
        proc_feat_mock.return_value = ('projections', 'comp')
        proc_comb_mock.return_value = ('projections', 'comp')

        process.process(data)

        proc_comb_mock.assert_called_with('desc', 'desc_color', 'rand_color',
                                          data.pca.config.descriptor_weight,
                                          data.pca.config.neutral_factor)
        self.assertEqual(0, proc_feat_mock.call_count)

    @patch('repstruct.analysis.pca.neutral_sub_pca_vector')
    @patch('repstruct.analysis.process.create_neutral_vector')
    def testProcessCombinedFeatures(self, neut_mock, pca_mock):
        neut_mock.return_value = 1
        pca_mock.return_value = ('proj', 'comp')

        descriptors = np.ones((1, 5))
        descriptors = descriptors / np.linalg.norm(descriptors)
        descriptor_colors = np.ones((1, 6))
        descriptor_colors = descriptor_colors / np.linalg.norm(descriptor_colors)
        random_colors = np.ones((1, 7))
        random_colors = random_colors / np.linalg.norm(random_colors)
        descriptor_weight = 0.5
        neutral_factor = 3

        process.process_combined_features(descriptors, descriptor_colors, random_colors,
                                          descriptor_weight, neutral_factor)

        neut_call_args = neut_mock.call_args[0]
        weight_sum = np.sum(neut_call_args[0][:, 1])
        self.assertLess(np.abs(weight_sum - 1), 0.00000001)

        pca_call_args = pca_mock.call_args[0]
        features_norm = np.linalg.norm(pca_call_args[0])
        self.assertLess(np.abs(features_norm - 1), 0.00000001)

    @patch('repstruct.analysis.kclosest.k_closest')
    def testClosest(self, close_mock):
        images = np.array([str(i) for i in range(0, 10)])
        pc_projections = np.ones((10, 3))

        data = dataset.DataSet('tag')
        data.pca.load = Mock(return_value=(images, pc_projections, 'components'))
        data.analysis = PropertyMock()
        data.analysis.config = PropertyMock()
        data.analysis.config.pc_projection_count = 2
        data.analysis.config.closest_group = 0.4
        data.analysis.config.representative = 0.1
        data.analysis.save_closest = Mock()

        closest_result = np.arange(0, int(data.analysis.config.closest_group * len(images)))
        representative_result = np.arange(0, int(data.analysis.config.representative * len(images)))
        close_mock.side_effect = [closest_result, representative_result]

        process.closest(data)

        call_args1 = close_mock.call_args_list[0][0]

        self.assertEqual(len(closest_result), call_args1[0])
        self.assertEqual(len(images), call_args1[1].shape[0])
        self.assertEqual(data.analysis.config.pc_projection_count, call_args1[1].shape[1])

        call_args2 = close_mock.call_args_list[1][0]

        self.assertEqual(len(representative_result), call_args2[0])
        self.assertEqual(len(closest_result), call_args2[1].shape[0])
        self.assertEqual(data.analysis.config.pc_projection_count, call_args1[1].shape[1])


if __name__ == '__main__':
    unittest.main()