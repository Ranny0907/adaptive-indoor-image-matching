import unittest

import numpy as np

try:
    from superglue_benchmark_lib import compute_reprojection_errors
except ModuleNotFoundError as exc:
    if exc.name != "cv2":
        raise
    compute_reprojection_errors = None


class ReprojectionErrorTest(unittest.TestCase):
    @unittest.skipIf(compute_reprojection_errors is None, "opencv-python is not installed")
    def test_compute_reprojection_errors_handles_opencv_point_shape(self):
        homography = np.array(
            [
                [1.0, 0.0, 10.0],
                [0.0, 1.0, -5.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        points0 = np.array(
            [
                [[0.0, 0.0]],
                [[10.0, 20.0]],
                [[-3.0, 4.0]],
            ],
            dtype=np.float32,
        )
        points1 = np.array(
            [
                [[10.0, -5.0]],
                [[20.0, 15.0]],
                [[7.0, -1.0]],
            ],
            dtype=np.float32,
        )

        errors = compute_reprojection_errors(points0, points1, homography)

        self.assertEqual(errors.shape, (3,))
        np.testing.assert_allclose(errors, np.zeros(3), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
