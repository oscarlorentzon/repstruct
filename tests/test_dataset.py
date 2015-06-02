import unittest
import numpy as np

from mock import Mock, patch

import repstruct.dataset as dataset

class TestDataSet(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def __assertProperties(self, instance):
        t = type(instance)
        property_names = [item for item in dir(t) if
                          isinstance(getattr(t, item), property)]

        new_value = 'new_value'
        for property_name in property_names:
            setattr(instance, property_name, new_value)

            self.assertEqual(new_value, getattr(instance, property_name))

    def testDataSet(self):
        tag = 'test_tag'
        data = dataset.DataSet(tag)

        self.assertEqual(tag, data.tag)

        self.assertTrue(type(data.collection) is dataset.CollectionDataSet)
        self.assertTrue(type(data.feature) is dataset.FeatureDataSet)
        self.assertTrue(type(data.descriptor) is dataset.DescriptorDataSet)
        self.assertTrue(type(data.pca) is dataset.PcaDataSet)
        self.assertTrue(type(data.analysis) is dataset.AnalysisDataSet)
        self.assertTrue(type(data.plot) is dataset.PlotDataSet)

        self.__assertProperties(data)

    @patch('os.makedirs')
    @patch('os.path.exists')
    def testDataSetBase(self, exists_mock, makedirs_mock):
        path = 'path'
        folder = 'folder'
        config = 'config'

        folder_path = path + '/' + folder

        exists_mock.return_value = False

        data = dataset.DataSetBase(path, folder, config)

        self.assertEqual(folder_path, data.path)
        self.assertEqual(config, data.config)

        self.__assertProperties(data)

        exists_mock.assert_called_with(folder_path)
        makedirs_mock.assert_called_with(folder_path)

    @patch('os.makedirs')
    @patch('os.path.exists')
    def testDataSetBaseNoPath(self, exists_mock, makedirs_mock):
        path = None
        folder = None
        config = None

        exists_mock.return_value = False

        data = dataset.DataSetBase(path, folder, config)

        self.assertEqual(None, data.path)
        self.assertEqual(None, data.config)

        self.assertEqual(0, exists_mock.call_count)
        self.assertEqual(0, makedirs_mock.call_count)

    @patch('os.path.isfile')
    @patch('os.listdir')
    @patch('os.path.join')
    def testCollectionDataSetImages(self, join_mock, listdir_mock, isfile_mock):
        data = dataset.CollectionDataSet(None, None)

        ims = ['im1.jpg, im2.jpg', 'no_im.txt']
        listdir_mock.return_value = ims

        join_mock.return_value = ''
        isfile_mock.return_value = True

        result = data.images()

        self.assertSequenceEqual(ims[:1], list(result))

    @patch('os.path.join')
    def testFeatureDataSetSave(self, join_mock):
        data = dataset.FeatureDataSet(None, None)

        join_mock.return_value = 'test.npz'

        with patch('numpy.savez') as savez_mock:
            locations = 'locations'
            descriptors = 'descriptors'
            data.save('none', locations, descriptors)

            savez_mock.assert_called_with(join_mock.return_value,
                                          locations=locations,
                                          descriptors=descriptors)

    @patch('os.path.join')
    def testFeatureDataSetLoad(self, join_mock):
        data = dataset.FeatureDataSet(None, None)

        join_mock.return_value = 'test.npz'

        with patch('numpy.load') as load_mock:
            load_mock.return_value = {'locations': 1, 'descriptors': 0}

            data.load('none')

            load_mock.assert_called_with(join_mock.return_value)


if __name__ == '__main__':
    unittest.main()
