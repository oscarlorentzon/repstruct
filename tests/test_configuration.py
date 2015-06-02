import unittest
import numpy as np

from mock import patch, MagicMock

import repstruct.configuration as configuration

class TestConfiguration(unittest.TestCase):

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

    @patch('yaml.load')
    @patch('os.path.join')
    def testConfigurationBase(self, join_mock, load_mock):
        configuration.ConfigurationBase(None, None)

        self.assertEqual(0, join_mock.call_count)
        self.assertEqual(0, load_mock.call_count)

    @patch('yaml.load')
    @patch('os.path.join')
    def testConfigurationBaseLoad(self, join_mock, load_mock):
        with patch('repstruct.configuration.open', create=True) as mock_open:
            mock_open.return_value = MagicMock(spec=file)

            configuration.ConfigurationBase('path', 'name')

        self.assertEqual(1, join_mock.call_count)
        self.assertEqual(1, load_mock.call_count)

    def testCollectionConfigurationEmpty(self):
        config = configuration.CollectionConfiguration()
        self.__assertProperties(config)

    @patch('yaml.load')
    @patch('os.path.join')
    def testCollectionConfigurationLoad(self, join_mock, load_mock):

        yml_config = {'processes': 2, 'collection_count': 3}
        load_mock.return_value = yml_config

        with patch('repstruct.configuration.open', create=True) as mock_open:
            mock_open.return_value = MagicMock(spec=file)

            config = configuration.CollectionConfiguration('path', 'name')

        self.assertEqual(yml_config['processes'], config.processes)
        self.assertEqual(yml_config['collection_count'], config.count)

    def testFeatureConfigurationEmpty(self):
        config = configuration.FeatureConfiguration()
        self.__assertProperties(config)

    @patch('yaml.load')
    @patch('os.path.join')
    def testFeatureConfigurationLoad(self, join_mock, load_mock):

        yml_config = {'edge_threshold': 2, 'peak_threshold': 3}
        load_mock.return_value = yml_config

        with patch('repstruct.configuration.open', create=True) as mock_open:
            mock_open.return_value = MagicMock(spec=file)

            config = configuration.FeatureConfiguration('path', 'name')

        self.assertEqual(yml_config['edge_threshold'], config.edge_threshold)
        self.assertEqual(yml_config['peak_threshold'], config.peak_threshold)

    def testPcaConfigurationEmpty(self):
        config = configuration.PcaConfiguration()
        self.__assertProperties(config)

    @patch('yaml.load')
    @patch('os.path.join')
    def testPcaConfigurationLoad(self, join_mock, load_mock):

        yml_config = {'descriptor_weight': 2, 'neutral_factor': 3, 'feature_mode': 'DESCRIPTORS'}
        load_mock.return_value = yml_config

        with patch('repstruct.configuration.open', create=True) as mock_open:
            mock_open.return_value = MagicMock(spec=file)

            config = configuration.PcaConfiguration('path', 'name')

        self.assertEqual(yml_config['descriptor_weight'], config.descriptor_weight)
        self.assertEqual(yml_config['neutral_factor'], config.neutral_factor)
        self.assertEqual(configuration.FeatureMode.Descriptors, config.feature_mode)

    @patch('yaml.load')
    @patch('os.path.join')
    def testPcaConfigurationFeatureMode(self, join_mock, load_mock):
        with patch('repstruct.configuration.open', create=True) as mock_open:
            mock_open.return_value = MagicMock(spec=file)

            load_mock.return_value = {'feature_mode': 'DESCRIPTORS'}
            config = configuration.PcaConfiguration('path', 'name')
            self.assertEqual(configuration.FeatureMode.Descriptors, config.feature_mode)

            load_mock.return_value = {'feature_mode': 'ALL'}
            config = configuration.PcaConfiguration('path', 'name')
            self.assertEqual(configuration.FeatureMode.All, config.feature_mode)

            load_mock.return_value = {'feature_mode': 'COLORS'}
            config = configuration.PcaConfiguration('path', 'name')
            self.assertEqual(configuration.FeatureMode.Colors, config.feature_mode)

    @patch('yaml.load')
    @patch('os.path.join')
    def testPcaConfigurationFeatureModeFaulty(self, join_mock, load_mock):
        with patch('repstruct.configuration.open', create=True) as mock_open:
            mock_open.return_value = MagicMock(spec=file)

            load_mock.return_value = {'feature_mode': 'FAULTY'}

            self.assertRaises(ValueError, configuration.PcaConfiguration, 'path', 'name')

    def testAnalysisConfigurationEmpty(self):
        config = configuration.AnalysisConfiguration()
        self.__assertProperties(config)

    @patch('yaml.load')
    @patch('os.path.join')
    def testAnalysisConfigurationLoad(self, join_mock, load_mock):

        yml_config = {'pc_projection_count': 10,
                      'closest_group': 4,
                      'representative': 2,
                      'clusters': 9,
                      'runs': 5,
                      'iterations': 3}
        load_mock.return_value = yml_config

        with patch('repstruct.configuration.open', create=True) as mock_open:
            mock_open.return_value = MagicMock(spec=file)

            config = configuration.AnalysisConfiguration('path', 'name')

        self.assertEqual(yml_config['pc_projection_count'], config.pc_projection_count)
        self.assertEqual(yml_config['closest_group'], config.closest_group)
        self.assertEqual(yml_config['representative'], config.representative)
        self.assertEqual(yml_config['clusters'], config.clusters)
        self.assertEqual(yml_config['runs'], config.runs)
        self.assertEqual(yml_config['iterations'], config.iterations)

    def testPlotConfigurationEmpty(self):
        config = configuration.PlotConfiguration()
        self.__assertProperties(config)

    @patch('yaml.load')
    @patch('os.path.join')
    def testPlotConfigurationLoad(self, join_mock, load_mock):

        pc_plots = np.array([2, 4])

        yml_config = {'save_plot': True,
                      'image_dimension': 4,
                      'columns': 2,
                      'ticks': True,
                      'pc_plots': [list(pc_plots + 1)]}
        load_mock.return_value = yml_config

        with patch('repstruct.configuration.open', create=True) as mock_open:
            mock_open.return_value = MagicMock(spec=file)

            config = configuration.PlotConfiguration('path', 'name')

        self.assertEqual(yml_config['save_plot'], config.save_plot)
        self.assertEqual(yml_config['image_dimension'], config.image_dimension)
        self.assertEqual(yml_config['columns'], config.columns)
        self.assertEqual(yml_config['ticks'], config.ticks)
        self.assertEqual(1, config.pc_plots.shape[0])
        self.assertSequenceEqual(list(pc_plots), list(config.pc_plots[0]))


if __name__ == '__main__':
    unittest.main()