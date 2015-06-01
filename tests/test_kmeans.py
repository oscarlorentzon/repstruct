import unittest
from mock import Mock, PropertyMock

from repstruct.analysis.kmeans import *
from repstruct.dataset import *


class TestKMeans(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testAllStructures(self):
        pc_projections = np.array([[.0, .0], [.1, .1], [.9, .9], [1., 1.]])

        data = DataSet('tag')
        data.pca = PropertyMock()
        data.pca.load = Mock(return_value=('', pc_projections, ''))
        data.analysis = PropertyMock
        data.analysis.config = PropertyMock()
        data.analysis.config.pc_projection_count = 2
        data.analysis.config.iterations = 100
        data.analysis.config.clusters = 2
        data.analysis.config.runs = 100
        data.analysis.save_structures = Mock()

        all_structures(data)

        structures = data.analysis.save_structures.call_args[0][1]

        self.assertEqual(2, structures.shape[0])

        for structure in structures:
            self.assertEqual(2, len(structure))

            if 0 in structure:
                self.assertTrue(1 in structure)
            else:
                self.assertTrue(2 in structure)
                self.assertTrue(3 in structure)

    def testScoreStructures(self):
        closest = (np.array([0, 1, 2, 3, 4]), np.array([0, 1]))
        structures = (np.array([0]), np.array([[0, 1], [2, 3], [4, 5]]))

        analysis_data = Mock()
        analysis_data.load_closest = Mock(return_value=closest)
        analysis_data.load_structures = Mock(return_value=structures)

        analysis_data.save_scored_structures = Mock()

        score_structures(analysis_data)

        call_args = analysis_data.save_scored_structures.call_args[0][0]

        for index, structure in enumerate(structures[1]):
            self.assertSequenceEqual(list(structure), list(call_args[index, :]))

    def testScoreStructuresInvert(self):
        closest = (np.array([1, 2, 3, 4, 5]), np.array([4, 5]))
        structures = (np.array([0]), np.array([[0, 1], [2, 3], [4, 5]]))

        analysis_data = Mock()
        analysis_data.load_closest = Mock(return_value=closest)
        analysis_data.load_structures = Mock(return_value=structures)

        analysis_data.save_scored_structures = Mock()

        score_structures(analysis_data)

        call_args = analysis_data.save_scored_structures.call_args[0][0]

        for index, structure in enumerate(structures[1][::-1]):
            self.assertSequenceEqual(list(structure), list(call_args[index, :]))

    def testScoreStructuresSameScore(self):
        closest = (np.array([0, 1, 2, 3, 4]), np.array([0, 2]))
        structures = (np.array([0]), np.array([[0, 1], [2, 3, 6], [4, 5]]))

        analysis_data = Mock()
        analysis_data.load_closest = Mock(return_value=closest)
        analysis_data.load_structures = Mock(return_value=structures)

        analysis_data.save_scored_structures = Mock()

        score_structures(analysis_data)

        call_args = analysis_data.save_scored_structures.call_args[0][0]

        for index, structure in enumerate(structures[1][np.array([1, 0, 2])]):
            self.assertSequenceEqual(list(structure), list(call_args[index]))


if __name__ == '__main__':
    unittest.main()