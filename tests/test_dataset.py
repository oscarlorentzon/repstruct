import unittest

import repstruct.dataset as dataset

class TestDataSet(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testDataSet(self):
        tag = 'test_tag'
        data = dataset.DataSet(tag)

        self.assertEqual(tag, data.tag)

        self.assertTrue(type(data.collection is dataset.CollectionDataSet))
        self.assertTrue(type(data.feature is dataset.FeatureDataSet))
        self.assertTrue(type(data.descriptor is dataset.DescriptorDataSet))
        self.assertTrue(type(data.pca is dataset.PcaDataSet))
        self.assertTrue(type(data.analysis is dataset.AnalysisDataSet))
        self.assertTrue(type(data.plot is dataset.PlotDataSet))

        property_names = [item for item in dir(dataset.DataSet) if
                          isinstance(getattr(dataset.DataSet, item), property)]

        new_value = 'new_value'
        for property_name in property_names:
            setattr(data, property_name, new_value)

            self.assertEqual(new_value, getattr(data, property_name))


if __name__ == '__main__':
    unittest.main()
