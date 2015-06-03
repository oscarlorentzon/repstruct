import unittest
import numpy as np
import repstruct.features.extract as extract
import repstruct.dataset as dataset

from mock import Mock, patch


class TestExtract(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testGetRgbFromLocations(self):
        im = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])

        row_locations = np.array([0])
        column_locations = np.array([0])

        rgb = extract.get_rgb_from_locs(row_locations, column_locations, im)

        self.assertEqual(3, len(rgb.shape))
        self.assertEqual(1, rgb.shape[0])
        self.assertEqual(1, rgb.shape[1])
        self.assertEqual(3, rgb.shape[2])

        self.assertEqual(im[0, 0, 0], rgb[0, 0, 0])
        self.assertEqual(im[0, 0, 1], rgb[0, 0, 1])
        self.assertEqual(im[0, 0, 2], rgb[0, 0, 2])

    def testGetRgbFromLocationsMultiple(self):
        im = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])

        row_locations = np.array([0, 1])
        column_locations = np.array([0, 1])

        rgb = extract.get_rgb_from_locs(row_locations, column_locations, im)

        self.assertEqual(3, len(rgb.shape))
        self.assertEqual(2, rgb.shape[0])
        self.assertEqual(1, rgb.shape[1])
        self.assertEqual(3, rgb.shape[2])

        self.assertEqual(im[0, 0, 0], rgb[0, 0, 0])
        self.assertEqual(im[0, 0, 1], rgb[0, 0, 1])
        self.assertEqual(im[0, 0, 2], rgb[0, 0, 2])

        self.assertEqual(im[1, 1, 0], rgb[1, 0, 0])
        self.assertEqual(im[1, 1, 1], rgb[1, 0, 1])
        self.assertEqual(im[1, 1, 2], rgb[1, 0, 2])

    def testGetRgbFromLocationsFloat(self):
        im = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])

        row_locations = np.array([0.4])
        column_locations = np.array([1.9])

        rgb = extract.get_rgb_from_locs(row_locations, column_locations, im)

        self.assertEqual(3, len(rgb.shape))
        self.assertEqual(1, rgb.shape[0])
        self.assertEqual(1, rgb.shape[1])
        self.assertEqual(3, rgb.shape[2])

        self.assertEqual(im[0, 1, 0], rgb[0, 0, 0])
        self.assertEqual(im[0, 1, 1], rgb[0, 0, 1])
        self.assertEqual(im[0, 1, 2], rgb[0, 0, 2])

    def testCreateNanArray(self):
        columns = 10

        result = extract.create_nan_array(columns)

        self.assertEqual(columns, result.shape[0])

        for value in result:
            self.assertTrue(np.isnan(value))

    def testRgbToHsCoordsBlackWhite(self):
        rgb = np.array([[[0, 0, 0]], [[255, 255, 255]]])

        result = extract.rgb_to_hs_coords(rgb)

        self.assertEqual(rgb.shape[0], result.shape[0])
        self.assertEqual(2, result.shape[1])

        self.assertLess(np.linalg.norm(result[0]), 0.0000001)
        self.assertLess(np.linalg.norm(result[1]), 0.0000001)

    def testRgbToHsCoorsSingleColors(self):
        rgb = np.array([[[255, 0, 0]], [[0, 0, 255]], [[0, 255, 0]]])

        result = extract.rgb_to_hs_coords(rgb)

        self.assertEqual(rgb.shape[0], result.shape[0])
        self.assertEqual(2, result.shape[1])

        self.assertLess(np.abs(np.linalg.norm(result[0]) - 1.), 0.0000001)
        self.assertLess(np.abs(np.linalg.norm(result[1]) - 1.), 0.0000001)
        self.assertLess(np.abs(np.linalg.norm(result[2]) - 1.), 0.0000001)

    @patch('repstruct.features.extract.get_color_hist')
    @patch('repstruct.features.descriptor.normalize_by_division')
    @patch('repstruct.features.descriptor.classify_cosine')
    @patch('repstruct.features.descriptor.normalize')
    @patch('cv2.imread')
    def testDescriptorExtractor(self, imread_mock, norm_mock, classify_mock, div_mock, hist_mock):
        imread_mock.return_value = np.array([[[0, 0, 0]]])

        desc_res = 'desc'
        color_res = ['color', 'random']
        div_mock.return_value = desc_res
        hist_mock.side_effect = color_res

        zero = np.zeros((1, 2))
        classify_mock.return_value = zero
        norm_mock.return_value = zero

        feature_data = dataset.FeatureDataSet(None, None)
        feature_data.load = Mock(return_value=(zero, zero))

        descriptor_data = dataset.DescriptorDataSet(None)
        descriptor_data.save = Mock()

        collection_data = dataset.CollectionDataSet(None, None)
        collection_data.path = 'path'

        desc_cc = np.array([0., 0.])
        desc_cc_norm = np.array([1., 1.])

        color_cc = np.array([0., 0.])
        color_cc_norm = np.array([1., 1.])

        x = np.array([0., 1.])
        y = np.array([0., 1.])

        extractor = extract.DescriptorExtractor(feature_data, descriptor_data, collection_data,
                                                desc_cc, desc_cc_norm, color_cc, color_cc_norm,
                                                x, y)

        im = 'im'
        extractor(im)

        descriptor_data.save.assert_called_with(im, desc_res, color_res[0], color_res[1])

    def testGetColorHist(self):
        im = np.array([[[0, 0, 0]], [[255, 0, 0]]], np.uint8)
        rows = np.array([0, 1])
        cols = np.array([0, 0])
        cluster_centers = np.array([[0., 0.], [1., 0.]])
        cluster_center_norm = np.ones(2, np.float)

        result = extract.get_color_hist(im, rows, cols, cluster_centers, cluster_center_norm)

        self.assertEqual(1, len(result.shape))
        self.assertEqual(2, result.shape[0])
        self.assertFalse(np.any(np.isnan(result)))

        norm = np.linalg.norm(result)
        self.assertLess(np.abs(norm - 1.), 0.0000001)

        # Same number of items in the two clusters.
        self.assertLess(np.abs(result[0] - result[1]), 0.0000001)

    def testGetColorHistGrayscale(self):
        im = np.array([[0], [255]], np.uint8)
        rows = np.array([0, 1])
        cols = np.array([0, 0])
        cluster_centers = np.array([[0., 0.], [1., 0.]])
        cluster_center_norm = np.ones(2, np.float)

        result = extract.get_color_hist(im, rows, cols, cluster_centers, cluster_center_norm)

        self.assertEqual(1, len(result.shape))
        self.assertEqual(2, result.shape[0])
        self.assertTrue(np.all(np.isnan(result)))


if __name__ == '__main__':
    unittest.main()
