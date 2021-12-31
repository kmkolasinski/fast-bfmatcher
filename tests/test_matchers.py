import unittest

import numpy as np

import fast_bfmatcher.matchers as matchers
from fast_bfmatcher.matching_ops import find_row_col_min_values, l2_distance_matrix
from fast_bfmatcher.utils import measuretime

np.random.seed(0)


def benchmark(name: str, method, steps: int = 500, warmup: int = 5):
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

    def test_blis_sgemm(self):

        from fast_bfmatcher.matching_ops import blis_sgemm_transpose, sgemm_transpose

        A = np.random.randn(1000, 128).astype(np.float32)
        B = np.random.randn(1000, 128).astype(np.float32)
        C = np.random.randn(1000, 1000).astype(np.float32) * 0

        blis_sgemm_transpose(1, A, B, 0.0, C)

        print(np.abs(C - A @ B.T).max())

        benchmark("cython blis", lambda: blis_sgemm_transpose(1, A, B, 0.0, C))
        benchmark("numpy", lambda: A @ B.T)
        benchmark("cython blas", lambda: sgemm_transpose(1, A, B, 0.0, C))

    def test_compute_distance_matrix(self):

        A = np.random.randn(1005, 128).astype(np.float32)
        B = np.random.randn(1000, 128).astype(np.float32)
        C = np.random.randn(1005, 1000).astype(np.float32)

        l2_distance_matrix(A, B, C)

        np_distance = matchers.NumpyBFL2Matcher.distance_matrix(A, B)

        error = np.abs(np_distance - C).max()
        print(f"L2 numpy / cython MAX error: {error}")
        benchmark("cython", lambda: l2_distance_matrix(A, B, C))
        benchmark("numpy", lambda: matchers.NumpyBFL2Matcher.distance_matrix(A, B))

    def test_find_row_col_min_values(self):
        C = np.random.randn(1005, 1000).astype(np.float32)

        row_indices, row_values, col_values = find_row_col_min_values(C)
        row_indices_np = C.argmin(1)
        row_values_np = C.min(1)
        col_values_np = C.min(0)

        print("row_indices error: ", np.abs(row_indices - row_indices_np).max())
        print("row_values error : ", np.abs(row_values - row_values_np).max())
        print("col_values error : ", np.abs(col_values - col_values_np).max())

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
