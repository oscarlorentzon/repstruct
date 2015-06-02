import unittest

from mock import Mock, patch

import repstruct.dataset as dataset
import repstruct.configuration as configuration

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

    @patch('numpy.savez')
    @patch('os.path.join')
    def testDataSetBaseSave(self, join_mock, save_mock):
        file_name = 'fn'
        argument = 'arg'

        join_mock.return_value = file_name + '.npz'

        data = dataset.DataSetBase(None, None, None)
        data._save(file_name, arg=argument)

        save_mock.assert_called_with(join_mock.return_value, arg=argument)

    @patch('numpy.load')
    @patch('os.path.join')
    def testDataSetBaseLoad(self, join_mock, load_mock):
        file_name = 'fn'

        join_mock.return_value = file_name + '.npz'

        data = dataset.DataSetBase(None, None, None)
        data._load(file_name)

        load_mock.assert_called_with(join_mock.return_value)

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

    def testFeatureDataSetSave(self):
        data = dataset.FeatureDataSet(None, None)
        data._save = Mock()

        im = 'im'
        locations = 'locations'
        descriptors = 'descriptors'
        data.save(im, locations, descriptors)

        data._save.assert_called_with(im + '.sift', locations=locations, descriptors=descriptors)

    def testFeatureDataSetLoad(self):
        data = dataset.FeatureDataSet(None, None)

        result = {'locations': 1, 'descriptors': 0}
        data._load = Mock(return_value=result)

        im = 'im'
        data.load(im)

        data._load.assert_called_with(im + '.sift')

    def testDescriptorDataSetSave(self):
        data = dataset.DescriptorDataSet(None)
        data._save = Mock()

        im = 'im'
        descriptors = 'descriptors'
        desc_colors = 'desc_colors'
        rand_colors = 'rand_colors'
        data.save(im, descriptors, desc_colors, rand_colors)

        data._save.assert_called_with(im + '.descriptors',
                                     descriptors=descriptors,
                                     descriptor_colors=desc_colors,
                                     random_colors=rand_colors)

    def testDescriptorDataSetLoad(self):
        data = dataset.DescriptorDataSet(None)

        im = 'im'
        result = {'descriptors': 0, 'descriptor_colors': 1, 'random_colors': 2}
        data._load = Mock(return_value=result)

        data.load(im)

        data._load.assert_called_with(im + '.descriptors')

    def testPcaDataSetSave(self):
        data = dataset.PcaDataSet(None, None)
        data._save = Mock()

        images = 'images'
        pc_projections = 'pc_projections'
        principal_components = 'principal_components'
        data.save(images, pc_projections, principal_components)

        data._save.assert_called_with('principal_components',
                                      images=images,
                                      pc_projections=pc_projections,
                                      principal_components=principal_components)

    def testPcaDataSetLoad(self):
        data = dataset.PcaDataSet(None, None)

        result = {'images': 0, 'pc_projections': 0, 'principal_components': 0}
        data._load = Mock(return_value=result)

        data.load()

        data._load.assert_called_with('principal_components')

    def testAnalysisDataSetSaveClosest(self):
        data = dataset.AnalysisDataSet(None, None)
        data._save = Mock()

        closest_group = 'closest_group'
        representative = 'representative'
        data.save_closest(closest_group, representative)

        data._save.assert_called_with('closest',
                                      closest_group=closest_group,
                                      representative=representative)

    def testAnalysisDataSetLoadClosest(self):
        data = dataset.AnalysisDataSet(None, None)

        result = {'closest_group': 0, 'representative': 1}
        data._load = Mock(return_value=result)

        data.load_closest()

        data._load.assert_called_with('closest')

    def testAnalysisDataSetSaveStructures(self):
        data = dataset.AnalysisDataSet(None, None)
        data._save = Mock()

        centroids = 'centroids'
        structures = 'structures'
        data.save_structures(centroids, structures)

        data._save.assert_called_with('structures',
                                      centroids=centroids,
                                      structures=structures)

    def testAnalysisDataSetLoadStrucutures(self):
        data = dataset.AnalysisDataSet(None, None)

        result = {'centroids': 0, 'structures': 1}
        data._load = Mock(return_value=result)

        data.load_structures()

        data._load.assert_called_with('structures')

    def testAnalysisDataSetSaveScoredStructures(self):
        data = dataset.AnalysisDataSet(None, None)
        data._save = Mock()

        scored_structures = 'scored_structures'
        data.save_scored_structures(scored_structures)

        data._save.assert_called_with('scored_structures',
                                      scored_structures=scored_structures)

    def testAnalysisDataSetLoadScoredStrucutures(self):
        data = dataset.AnalysisDataSet(None, None)

        result = {'scored_structures': 0}
        data._load = Mock(return_value=result)

        data.load_scored_structures()

        data._load.assert_called_with('scored_structures')

    def testCollectionDataSetConfigType(self):
        data = dataset.CollectionDataSet(None, None)

        self.assertTrue(type(data.config) is configuration.CollectionConfiguration)

    def testFeatureDataSetConfigType(self):
        data = dataset.FeatureDataSet(None, None)

        self.assertTrue(type(data.config) is configuration.FeatureConfiguration)

    def testDescriptorDataSetConfigType(self):
        data = dataset.DescriptorDataSet(None)

        self.assertTrue(type(data.config) is dict)

    def testPcaDataSetConfigType(self):
        data = dataset.PcaDataSet(None, None)

        self.assertTrue(type(data.config) is configuration.PcaConfiguration)

    def testAnalysisDataSetConfigType(self):
        data = dataset.AnalysisDataSet(None, None)

        self.assertTrue(type(data.config) is configuration.AnalysisConfiguration)

    def testPlotDataSetConfigType(self):
        data = dataset.PlotDataSet(None, None)

        self.assertTrue(type(data.config) is configuration.PlotConfiguration)


if __name__ == '__main__':
    unittest.main()
