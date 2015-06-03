import unittest
from mock import Mock, patch, PropertyMock

import cv2

import repstruct.features.sift as sift
from repstruct.dataset import *


class TestSift(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testExtractSiftFeatures(self):
        im_size = 50
        box_size = 10
        box_low = im_size/2-box_size/2
        box_high = im_size/2+box_size/2
        im = np.zeros((im_size, im_size), np.uint8)
        im[box_low:box_high, box_low:box_high] = 255 * np.ones((box_size, box_size), np.uint8)

        locations, descriptors = sift.extract_sift_features(im)

        self.assertEqual(4, locations.shape[1])

        self.assertTrue(np.min(locations[:, 0]) > 0)
        self.assertTrue(np.max(locations[:, 0]) < 50)

        self.assertTrue(np.min(locations[:, 1]) > 0)
        self.assertTrue(np.max(locations[:, 1]) < 50)

        self.assertTrue(np.min(locations[:, 2]) > 0)
        self.assertTrue(np.min(locations[:, 3]) > 0)

        self.assertEqual(locations.shape[0], descriptors.shape[0])
        self.assertEqual(128, descriptors.shape[1])

    @patch('repstruct.features.sift.SiftExtractor')
    @patch('multiprocessing.Pool')
    def testExtract(self, mock_pool, mock_sift):
        data = DataSet('tag')
        data.feature = PropertyMock()
        data.collection = PropertyMock()
        images = np.array(['im1', 'im2'])
        data.collection.images = Mock(return_value=images)
        data.collection.config = PropertyMock()
        data.collection.config.processes = 1

        sift_instance = mock_sift.return_value
        pool_instance = mock_pool.return_value
        pool_instance.map = Mock(return_value=0)
        sift.extract(data)

        self.assertEqual(len(images), sift_instance.call_count)
        self.assertEqual(0, pool_instance.map.call_count)

    @patch('repstruct.features.sift.SiftExtractor')
    @patch('multiprocessing.Pool')
    def testExtractMultipleProcesses(self, mock_pool, mock_sift):
        data = DataSet('tag')
        data.feature = PropertyMock()
        data.collection = PropertyMock()
        images = np.array(['im1', 'im2'])
        data.collection.images = Mock(return_value=images)
        data.collection.config = PropertyMock()
        data.collection.config.processes = 2

        sift_instance = mock_sift.return_value
        pool_instance = mock_pool.return_value
        pool_instance.map = Mock(return_value=0)
        sift.extract(data)

        self.assertEqual(0, sift_instance.call_count)
        self.assertEqual(1, pool_instance.map.call_count)

    @patch('os.remove')
    @patch('cv2.imread')
    def testSiftExtractor(self, imread_mock, remove_mock):


        feature_data = FeatureDataSet(None, None)
        feature_data.save = Mock()

        collection_data = CollectionDataSet(None, None)
        collection_data._path = 'path'

        im_size = 50
        box_size = 10
        box_low = im_size/2-box_size/2
        box_high = im_size/2+box_size/2
        im = np.zeros((im_size, im_size), np.uint8)
        im[box_low:box_high, box_low:box_high] = 255 * np.ones((box_size, box_size), np.uint8)
        imread_mock.return_value = im

        extractor = sift.SiftExtractor(feature_data, collection_data)

        locations = [1, 1, 0, 0]
        descriptors = [0, 1, 2, 3]
        sift.extract_sift_features = Mock(return_value=(np.array([locations]), np.array([descriptors])))

        im_name = 'im_name'
        extractor(im_name)

        call_args = feature_data.save.call_args[0]

        self.assertEqual(im_name, call_args[0])
        self.assertSequenceEqual(locations, list(call_args[1][0]))
        self.assertSequenceEqual(list(descriptors), list(call_args[2][0]))

        self.assertEqual(0, remove_mock.call_count)

    @patch('os.remove')
    @patch('cv2.imread')
    def testSiftExtractorCorruptImage(self, imread_mock, remove_mock):
        imread_mock.return_value = None

        collection_data = CollectionDataSet(None, None)
        collection_data._path = 'path'

        extractor = sift.SiftExtractor(None, collection_data)
        im_name = 'im_name'
        extractor(im_name)

        self.assertEqual(1, remove_mock.call_count)


if __name__ == '__main__':
    unittest.main()