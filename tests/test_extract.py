import unittest
import numpy as np
from repstruct.features.extract import get_rgb_from_locs


class TestExtract(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testGetRgbFromLocations(self):
        im = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])

        row_locations = np.array([0])
        column_locations = np.array([0])

        rgb = get_rgb_from_locs(row_locations, column_locations, im)

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

        rgb = get_rgb_from_locs(row_locations, column_locations, im)

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

        rgb = get_rgb_from_locs(row_locations, column_locations, im)

        self.assertEqual(3, len(rgb.shape))
        self.assertEqual(1, rgb.shape[0])
        self.assertEqual(1, rgb.shape[1])
        self.assertEqual(3, rgb.shape[2])

        self.assertEqual(im[0, 1, 0], rgb[0, 0, 0])
        self.assertEqual(im[0, 1, 1], rgb[0, 0, 1])
        self.assertEqual(im[0, 1, 2], rgb[0, 0, 2])


if __name__ == '__main__':
    unittest.main()
