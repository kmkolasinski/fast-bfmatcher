import unittest

import numpy as np

import fast_bfmatcher.matchers as matchers
from fast_bfmatcher.matching_ops import l2_distance_matrix
from fast_bfmatcher.utils import measuretime

np.random.seed(0)


def benchmark(name: str, method, steps: int = 500, warmup: int = 50):
    print()
    with measuretime(f"{name} warmup", num_steps=steps):
        for _ in range(warmup):
            method()

    with measuretime(f"{name} calls ", num_steps=steps):
        for _ in range(steps):
            method()


class TestMatching(unittest.TestCase):
    def setUp(self) -> None:
        self.X = np.random.randint(0, 128, (512, 256), dtype=np.uint8)

    def test_run_blas_dot(self):

        A = np.random.randn(1005, 128).astype(np.float32)
        B = np.random.randn(1000, 128).astype(np.float32)
        C = np.random.randn(1005, 1000).astype(np.float32)

        l2_distance_matrix(A, B, C)

        np_distance = matchers.NumpyBFL2Matcher.distance_matrix(A, B)

        error = np.abs(np_distance - C).max()
        print(f"L2 numpy / cython MAX error: {error}")
        benchmark("cython", lambda: l2_distance_matrix(A, B, C))
        benchmark("numpy", lambda: matchers.NumpyBFL2Matcher.distance_matrix(A, B))

    #
    # def test_mean_square_cols(self):
    #
    #     C = np.random.randn(1005, 1000).astype(np.float32)
    #
    #     row_values = np.zeros([C.shape[0]], dtype=np.float32)
    #     mean_square_cols(C, row_values)
    #     error = row_values - np.mean(C**2, 1)
    #     print("error:", np.max(error))
    #
    #     blas = lambda: mean_square_cols(C, row_values)
    #     np_call = lambda : np.mean(C**2, 1)
    #
    #     print()
    #     # benchmark("blas1", blas_call)
    #     benchmark("blas", blas)
    #     benchmark("np", np_call)
    #     print()
    #
    # def test_argmin_col(self):
    #
    #     C = np.random.randn(1005, 1003).astype(np.float32)
    #
    #     row_indices = np.zeros([C.shape[0]], dtype=np.int32)
    #     column_argmin(C, row_indices)
    #     error = row_indices - np.argmin(C, 1)
    #     print("error:", np.max(error))
    #
    #     blas = lambda: column_argmin(C, row_indices)
    #     np_call = lambda : np.argmin(C, 1)
    #
    #     print()
    #     # benchmark("blas1", blas_call)
    #     benchmark("blas", blas)
    #     benchmark("np", np_call)
    #     print()
    #
    # def test_blas_match(self):
    #
    #     A = np.random.randn(1000, 128).astype(np.float32)
    #     B = np.random.randn(900, 128).astype(np.float32)
    #     C = np.random.randn(1000, 900).astype(np.float32)
    #
    #     l2_distance_matrix(A, B, C)
    #     row_indices = np.zeros([A.shape[0]], dtype=np.int32)
    #     col_indices = np.zeros([B.shape[0]], dtype=np.int32)
    #     argmin_match(C, row_indices, col_indices)
    #
    #     dist = distance_matrix(A, B)
    #
    #     print(np.abs( np.argmin(dist, 1) - row_indices).max())
    #     print(np.abs( np.argmin(dist, 0) - col_indices).max())

    def test_matchers(self):
        X = np.random.randint(0, 255, size=(1000, 128)).astype(np.float32)
        Y = np.random.randint(0, 255, size=(1005, 128)).astype(np.float32)

        fast_matcher = matchers.FastBFL2Matcher()
        result = fast_matcher.match(X, Y)

        cv_matcher = matchers.CVBFL2Matcher()
        cv_result = cv_matcher.match(X, Y)

        self.assertEqual(result, cv_result)

        np_matcher = matchers.NumpyBFL2Matcher()
        np_result = np_matcher.match(X, Y)

        self.assertEqual(np_result, cv_result)

        benchmark("fast", lambda: fast_matcher.match(X, Y))
        benchmark("opencv", lambda: cv_matcher.match(X, Y))
        benchmark("numpy", lambda: np_matcher.match(X, Y))

        try:
            tf_matcher = matchers.TFL2BFMatcher()
            tf_result = tf_matcher.match(X, Y)

            np.testing.assert_equal(tf_result.indices, cv_result.indices)
            max_error = np.abs(tf_result.distances - cv_result.distances).max()
            self.assertLess(max_error, 0.001)

            benchmark("tensorflow", lambda: tf_matcher.match(X, Y))
        except Exception:
            pass
