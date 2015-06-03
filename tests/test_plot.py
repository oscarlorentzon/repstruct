import unittest
import math
import numpy as np

from mock import patch

import repstruct.display.plot as plot

import matplotlib.pyplot as pl


class TestPlot(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @patch('cv2.imwrite')
    @patch('cv2.resize')
    @patch('cv2.imread')
    def testPlotRepresentativeOneCol(self, imread_mock, resize_mock, imwrite_mock):
        im_dim = 10
        image_count = 3
        closest = 2
        representative = 1

        imread_mock.return_value = np.zeros((im_dim, im_dim, 3), np.uint8)

        def side_effect(im, dsize):
            return np.zeros((dsize[0], dsize[1], 3), np.uint8)

        resize_mock.side_effect = side_effect

        images = np.array([str(i) for i in range(0, image_count)])
        index_closest = np.array(range(0, closest))
        index_representative = np.array(range(0, representative))

        with patch('os.path.join') as join_mock:
            join_mock.return_value = 'file'
            plot.plot_representative('dir', images, index_closest, index_representative,
                                     im_dim=im_dim, cols=1, save_path='path')

        im_result = imwrite_mock.call_args[0][1]

        self.assertEqual(3, len(im_result.shape))

        space = im_dim / 10
        border = 2 * space
        width = im_dim + 2 * border

        self.assertEqual(width, im_result.shape[1])

        count = 0
        for row in im_result:
            for pixel in row:
                count = count if np.any(pixel) else count + 1

        space_count = 2 * im_dim * space
        im_count = count - space_count

        self.assertEqual((image_count + closest + representative) * im_dim**2, im_count)

    @patch('cv2.imwrite')
    @patch('cv2.resize')
    @patch('cv2.imread')
    def testPlotRepresentativeMultipleCols(self, imread_mock, resize_mock, imwrite_mock):
        im_dim = 20
        cols = 5
        image_count = 25
        closest = 8
        representative = 3

        imread_mock.return_value = np.zeros((im_dim, im_dim, 3), np.uint8)

        def side_effect(im, dsize):
            return np.zeros((dsize[0], dsize[1], 3), np.uint8)

        resize_mock.side_effect = side_effect

        images = np.array([str(i) for i in range(0, image_count)])
        index_closest = np.array(range(0, closest))
        index_representative = np.array(range(0, representative))

        with patch('os.path.join') as join_mock:
            join_mock.return_value = 'file'
            plot.plot_representative('dir', images, index_closest, index_representative,
                                     im_dim=im_dim, cols=cols, save_path='path')

        im_result = imwrite_mock.call_args[0][1]

        self.assertEqual(3, len(im_result.shape))

        space = im_dim / 10
        border = 2 * space
        width = cols * im_dim + (cols - 1) * space + 2 * border

        self.assertEqual(width, im_result.shape[1])

        count = 0
        for row in im_result:
            for pixel in row:
                count = count if np.any(pixel) else count + 1

        space_count = 2 * (width - 2 * border) * space
        im_count = count - space_count

        self.assertEqual((image_count + closest) * im_dim**2 + representative * (2*im_dim)**2, im_count)

    @patch('matplotlib.pyplot.axis')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.imshow')
    @patch('matplotlib.pyplot.figure')
    @patch('cv2.resize')
    @patch('cv2.imread')
    def testPlotRepresentativePlot(self, imread_mock, resize_mock, figure_mock, imshow_mock, show_mock, axis_mock):
        im_dim = 10

        imread_mock.return_value = np.zeros((im_dim, im_dim, 3), np.uint8)

        def side_effect(im, dsize):
            return np.zeros((dsize[0], dsize[1], 3), np.uint8)

        resize_mock.side_effect = side_effect

        images = np.array([str(i) for i in range(0, 3)])
        index_closest = np.array(range(0, 2))
        index_representative = np.array(range(0, 1))

        with patch('os.path.join') as join_mock:
            join_mock.return_value = 'file'
            plot.plot_representative('dir', images, index_closest, index_representative,
                                     im_dim=im_dim, cols=1)

        self.assertEqual(1, imshow_mock.call_count)
        self.assertEqual(1, figure_mock.call_count)
        self.assertEqual(1, axis_mock.call_count)
        self.assertEqual(1, show_mock.call_count)

    @patch('cv2.imwrite')
    @patch('cv2.resize')
    @patch('cv2.imread')
    def testPlotStructuresOneCol(self, imread_mock, resize_mock, imwrite_mock):
        im_dim = 10
        image_count = 5
        structures = np.array([[0, 1], [2, 3], [4]])

        imread_mock.return_value = np.zeros((im_dim, im_dim, 3), np.uint8)

        def side_effect(im, dsize):
            return np.zeros((dsize[0], dsize[1], 3), np.uint8)

        resize_mock.side_effect = side_effect

        images = np.array([str(i) for i in range(0, image_count)])

        with patch('os.path.join') as join_mock:
            join_mock.return_value = 'file'
            plot.plot_structures('dir', images, structures,
                                 im_dim=im_dim, cols=1, save_path='path')

        im_result = imwrite_mock.call_args[0][1]

        self.assertEqual(3, len(im_result.shape))

        space = im_dim / 10
        border = 2 * space
        width = im_dim + 2 * border

        self.assertEqual(width, im_result.shape[1])

        count = 0
        for row in im_result:
            for pixel in row:
                count = count if np.any(pixel) else count + 1

        space_count = (structures.shape[0] - 1) * im_dim * space
        im_count = count - space_count

        self.assertEqual(image_count * im_dim**2, im_count)

    @patch('cv2.imwrite')
    @patch('cv2.resize')
    @patch('cv2.imread')
    def testPlotStructuresMultipleCols(self, imread_mock, resize_mock, imwrite_mock):
        im_dim = 100
        cols = 2
        image_count = 5
        structures = np.array([[0, 1], [2, 3], [4]])

        imread_mock.return_value = np.zeros((im_dim, im_dim, 3), np.uint8)

        def side_effect(im, dsize):
            return np.zeros((dsize[0], dsize[1], 3), np.uint8)

        resize_mock.side_effect = side_effect

        images = np.array([str(i) for i in range(0, image_count)])

        with patch('os.path.join') as join_mock:
            join_mock.return_value = 'file'
            plot.plot_structures('dir', images, structures,
                                 im_dim=im_dim, cols=cols, save_path='path')

        im_result = imwrite_mock.call_args[0][1]

        self.assertEqual(3, len(im_result.shape))

        space = im_dim / 10
        border = 2 * space
        width = cols * im_dim + (cols - 1) * space + 2 * border

        self.assertEqual(width, im_result.shape[1])

        count = 0
        for row in im_result:
            for pixel in row:
                count = count if np.any(pixel) else count + 1

        space_count = (structures.shape[0] - 1) * (width - 2 * border) * space
        im_count = count - space_count

        self.assertEqual(image_count * im_dim**2, im_count)

    @patch('matplotlib.pyplot.axis')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.imshow')
    @patch('matplotlib.pyplot.figure')
    @patch('cv2.resize')
    @patch('cv2.imread')
    def testPlotStructuresPlot(self, imread_mock, resize_mock, figure_mock, imshow_mock, show_mock, axis_mock):
        im_dim = 10
        image_count = 5
        structures = np.array([[0, 1], [2, 3], [4]])

        imread_mock.return_value = np.zeros((im_dim, im_dim, 3), np.uint8)

        def side_effect(im, dsize):
            return np.zeros((dsize[0], dsize[1], 3), np.uint8)

        resize_mock.side_effect = side_effect

        images = np.array([str(i) for i in range(0, image_count)])

        with patch('os.path.join') as join_mock:
            join_mock.return_value = 'file'
            plot.plot_structures('dir', images, structures,
                                 im_dim=im_dim, cols=1)

        self.assertEqual(1, imshow_mock.call_count)
        self.assertEqual(1, figure_mock.call_count)
        self.assertEqual(1, axis_mock.call_count)
        self.assertEqual(1, show_mock.call_count)


if __name__ == '__main__':
    unittest.main()
