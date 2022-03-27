import unittest

import numpy as np

import fast_bfmatcher.matchers as matchers
from fast_bfmatcher.extra.cv import OpenCVL2CCBFMatcher, OpenCVL2LoweBFMatcher
from fast_bfmatcher.extra.np import NumpyL2CCBFMatcher
from fast_bfmatcher.matching_ops import (
    find_lowes_test_matches,
    find_row_col_min_values,
    l2_distance_matrix,
)
from fast_bfmatcher.utils import measuretime

np.random.seed(0)


def benchmark(name: str, method, steps: int = 50, warmup: int = 5):

    for _ in range(warmup):
        method()

    with measuretime(f"{name} calls ", num_steps=steps):
        for _ in range(steps):
            method()


class TestMatching(unittest.TestCase):
    def test_blis_sgemm(self):

        from fast_bfmatcher.matching_ops import (
            blas_sgemm_transpose,
            blis_sgemm_transpose,
        )

        A = np.random.randn(1000, 128).astype(np.float32)
        B = np.random.randn(1000, 128).astype(np.float32)
        C = np.random.randn(1000, 1000).astype(np.float32) * 0

        blis_sgemm_transpose(1, A, B, 0.0, C)

        print("Error:", np.abs(C - A @ B.T).max())

        benchmark("cython blis", lambda: blis_sgemm_transpose(1, A, B, 0.0, C))
        benchmark("numpy", lambda: A @ B.T)
        benchmark("cython blas", lambda: blas_sgemm_transpose(1, A, B, 0.0, C))

    def test_compute_distance_matrix(self):

        A = np.random.randn(1005, 128).astype(np.float32)
        B = np.random.randn(1000, 128).astype(np.float32)
        C = np.random.randn(1005, 1000).astype(np.float32)

        l2_distance_matrix(A, B, C)

        np_distance = NumpyL2CCBFMatcher.l2_distance_matrix(A, B)

        error = np.abs(np_distance - C).max()
        print(f"L2 numpy / cython MAX error: {error}")
        benchmark("cython", lambda: l2_distance_matrix(A, B, C))
        benchmark("numpy", lambda: NumpyL2CCBFMatcher.l2_distance_matrix(A, B))

    def test_find_row_col_min_values(self):
        C = np.random.randn(2000, 2000).astype(np.float32)

        row_indices, row_values, col_values = find_row_col_min_values(C)

        def _find_row_col_min_values_np(x):
            row_indices_np = x.argmin(1)
            row_values_np = x.min(1)
            col_values_np = x.min(0)
            return row_indices_np, row_values_np, col_values_np

        row_indices_np, row_values_np, col_values_np = _find_row_col_min_values_np(C)

        print("row_indices error: ", np.abs(row_indices - row_indices_np).max())
        print("row_values error : ", np.abs(row_values - row_values_np).max())
        print("col_values error : ", np.abs(col_values - col_values_np).max())

        benchmark("cython", lambda: find_row_col_min_values(C))
        benchmark("numpy", lambda: _find_row_col_min_values_np(C))

    def test_find_lowe_matches(self):
        C = np.random.rand(100, 101).astype(np.float32)
        ratio = 0.7
        row_indices, row_values = find_lowes_test_matches(C, ratio=ratio)

        for i in range(20):
            indices = np.argsort(C[i])
            d1, d2 = C[i][indices][:2]
            if d1 / d2 > ratio:
                self.assertEqual(row_indices[i], -1)
                self.assertEqual(row_values[i], 0)
            else:
                self.assertEqual(row_indices[i], indices[0])
                self.assertEqual(row_values[i], d1)

    def test_matchers(self):
        X = np.random.randint(0, 255, size=(1000, 128)).astype(np.float32)
        Y = np.random.randint(0, 255, size=(1005, 128)).astype(np.float32)

        fast_matcher = matchers.FastL2CCBFMatcher()
        result = fast_matcher.match(X, Y)

        cv_matcher = OpenCVL2CCBFMatcher()
        cv_result = cv_matcher.match(X, Y)

        self.assertEqual(result, cv_result)

        np_matcher = NumpyL2CCBFMatcher()
        np_result = np_matcher.match(X, Y)

        self.assertEqual(np_result, cv_result)

        benchmark("fast", lambda: fast_matcher.match(X, Y))
        benchmark("opencv", lambda: cv_matcher.match(X, Y))
        benchmark("numpy", lambda: np_matcher.match(X, Y))

        try:
            from fast_bfmatcher.extra.tf import TFL2CCBFMatcher

            tf_matcher = TFL2CCBFMatcher()
            tf_result = tf_matcher.match(X, Y)

            np.testing.assert_equal(tf_result.indices, cv_result.indices)
            max_error = np.abs(tf_result.distances - cv_result.distances).max()
            self.assertLess(max_error, 0.001)

            benchmark("tensorflow", lambda: tf_matcher.match(X, Y))
        except Exception:
            pass

    def test_benchmark(self):
        from fast_bfmatcher.benchmark import benchmark_cc_matchers

        benchmark_cc_matchers(1, 1, num_kpts=1000)
        benchmark_cc_matchers(20, 10, num_kpts=1000)

    def test_lowes_test_matchers(self):
        X = np.random.randint(0, 255, size=(105, 128)).astype(np.float32)
        Y = np.random.randint(0, 255, size=(100, 128)).astype(np.float32)

        Y[0, :] = X[10, :]

        ratio = 0.7
        fast_matcher = matchers.FastL2LoweBFMatcher(ratio)
        cv_matcher = OpenCVL2LoweBFMatcher(ratio)
        fast_result = fast_matcher.match(X, Y)

        cv_result = cv_matcher.match(X, Y)

        self.assertEqual(fast_result, cv_result)
