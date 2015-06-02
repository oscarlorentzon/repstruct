import unittest
import numpy as np
import repstruct.features.extract as extract


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


if __name__ == '__main__':
    unittest.main()
