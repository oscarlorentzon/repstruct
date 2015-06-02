import unittest
import numpy as np

import repstruct.analysis.pca as pca
import repstruct.analysis.process as process


class TestPca(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testNeutralSubPcaZeroNeutFactor(self):
        x = np.array([[2., 0.], [0., 1.]])
        neut_factor = 0

        projections, components = pca.neutral_sub_pca(x, neut_factor)

        # Components should be normalized
        norm = np.linalg.norm(components, axis=0)
        max_norm_error = np.max(np.abs(norm - 1.0))

        self.assertLess(max_norm_error, 0.0000001, 'The norm is not one for the normalized array.')

        c = np.linalg.norm(np.subtract(np.eye(2), components))
        self.assertLess(c, 0.0000001)

        proj_norm_diff = np.abs(np.linalg.norm(x, axis=1) - np.linalg.norm(projections, axis=1))
        max_proj_norm_diff = np.max(np.abs(norm - 1.0))

        self.assertLess(max_proj_norm_diff, 0.0000001)

        p = np.linalg.norm(np.subtract(x, projections))
        self.assertLess(p, 0.0000001)

    def testNeutralSubPca(self):
        x = np.array([[2., 0.], [0., 1.]])
        neutral_vector = process.create_neutral_vector(np.array([[2, 1.]]), 2)
        x_n = x + neutral_vector
        neut_factor = 1.

        projections, components = pca.neutral_sub_pca(x_n, neut_factor)

        c = np.linalg.norm(np.subtract(np.eye(2), components))
        self.assertLess(c, 0.0000001)

        p = np.linalg.norm(np.subtract(x, projections))
        self.assertLess(p, 0.0000001)

    def testNeutralSubPcaVectorEmpty(self):
        x = np.array([[2., 0.], [0., 1.]])
        neutral_vector = np.zeros((2, 2), dtype=np.float)

        projections, components = pca.neutral_sub_pca_vector(x, neutral_vector)

        # Components should be normalized
        norm = np.linalg.norm(components, axis=0)
        max_norm_error = np.max(np.abs(norm - 1.0))

        self.assertLess(max_norm_error, 0.0000001, 'The norm is not one for the normalized array.')

        c = np.linalg.norm(np.subtract(np.eye(2), components))
        self.assertLess(c, 0.0000001)

        p = np.linalg.norm(np.subtract(x, projections))
        self.assertLess(p, 0.0000001)

    def testNeutralSubPcaVector(self):
        x = np.array([[2., 0.], [0., 1.]])
        neutral_vector = process.create_neutral_vector(np.array([[2, 1.]]), 2)
        x_n = x + neutral_vector

        projections, components = pca.neutral_sub_pca_vector(x_n, neutral_vector)

        c = np.linalg.norm(np.subtract(np.eye(2), components))
        self.assertLess(c, 0.0000001)

        p = np.linalg.norm(np.subtract(x, projections))
        self.assertLess(p, 0.0000001)


if __name__ == '__main__':
    unittest.main()
