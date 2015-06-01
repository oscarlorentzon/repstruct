import unittest
import math

from repstruct.analysis.kclosest import *
from repstruct.dataset import *


class TestKClosest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testKClosest(self):
        k = 2
        angles = [.0, .1, 1., 2.]
        vs = []
        for angle in angles:
            vs.append([math.cos(angle), math.sin(angle)])

        vs = np.array(vs)

        result = k_closest(k, vs)

        self.assertSequenceEqual([0, 1], sorted(list(result)))

    def testKClosestLargerResultSet(self):
        k = 3
        angles = [.0, .1, .9, 1., 1.1, 2., 3.]
        vs = []
        for angle in angles:
            vs.append([math.cos(angle), math.sin(angle)])

        vs = np.array(vs)

        result = k_closest(k, vs)

        self.assertSequenceEqual([2, 3, 4], sorted(list(result)))

    def testKClosestLargerRandomOrder(self):
        k = 3
        angles = np.array([.9, .0, 3., .1, 1.1, 2., 1.])
        vs = []
        for angle in angles:
            vs.append([math.cos(angle), math.sin(angle)])

        vs = np.array(vs)

        result = k_closest(k, vs)

        self.assertSequenceEqual([0, 4, 6], sorted(list(result)))

    def testKClosestLargerEuclidean(self):
        k = 3
        angles = [.0, .1, .9, 1., 1.1, 2., 3.]
        vs = []
        for angle in angles:
            vs.append([math.cos(angle), math.sin(angle)])

        vs = np.array(vs)

        result = k_closest(k, vs, 'euclidean')

        self.assertSequenceEqual([2, 3, 4], sorted(list(result)))


if __name__ == '__main__':
    unittest.main()