import os
import unittest
from pathlib import Path

import cv2
import numpy as np

import fast_bfmatcher as matchers
import fast_bfmatcher.matching_ops as mops
from fast_bfmatcher.extra.cv import OpenCVL2CCBFMatcher, OpenCVL2RTBFMatcher
from fast_bfmatcher.extra.np import NumpyL2CCBFMatcher
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

        A = np.random.randn(1000, 128).astype(np.float32)
        B = np.random.randn(1000, 128).astype(np.float32)
        C = np.random.randn(1000, 1000).astype(np.float32) * 0

        mops.blis_sgemm_transpose(1, A, B, 0.0, C)
        error = np.abs(C - A @ B.T).max()
        error = error / C.max()
        print("Multiplication error:", error)
        self.assertLess(error, 1e-6)

        benchmark("blis", lambda: mops.blis_sgemm_transpose(1, A, B, 0.0, C))
        benchmark("numpy", lambda: A @ B.T)

    def test_compute_distance_matrix(self):

        A = np.random.randn(1005, 128).astype(np.float32)
        B = np.random.randn(1000, 128).astype(np.float32)
        C = np.random.randn(1005, 1000).astype(np.float32)

        mops.l2_distance_matrix(A, B, C)

        np_distance = NumpyL2CCBFMatcher.l2_distance_matrix(A, B)

        error = np.abs(np_distance - C).max() / np_distance.max()
        print(f"L2 numpy / cython MAX error: {error}")
        self.assertLess(error, 1e-6)
        benchmark("cython", lambda: mops.l2_distance_matrix(A, B, C))
        benchmark("numpy", lambda: NumpyL2CCBFMatcher.l2_distance_matrix(A, B))

    def test_find_row_col_min_values(self):
        C = np.random.randn(2000, 2000).astype(np.float32)

        row_indices, row_values = mops.find_cross_check_matches(C)

        def _find_row_col_min_values_np(x):
            row_indices_np = x.argmin(1)
            row_values_np = x.min(1)
            return row_indices_np, row_values_np

        row_indices_np, row_values_np = _find_row_col_min_values_np(C)
        mask = np.array(row_indices) > -1

        indices_error = np.abs(row_indices - row_indices_np)[mask].max()
        values_error = np.abs(row_values - row_values_np)[mask].max()

        print("indices error: ", indices_error)
        print("values error : ", values_error)
        print("num matches : ", mask.sum())

        self.assertEqual(indices_error, 0)
        self.assertEqual(values_error, 0)

        benchmark("cython", lambda: mops.find_cross_check_matches(C))
        benchmark("numpy", lambda: _find_row_col_min_values_np(C))

    def test_find_ratio_test_matches(self):
        C = np.random.rand(100, 101).astype(np.float32)
        ratio = 0.7
        row_indices, row_values = mops.find_ratio_test_matches(C, ratio=ratio)

        for i in range(100):
            indices = np.argsort(C[i])
            d1, d2 = C[i][indices][:2]
            if d1 / d2 > ratio:
                self.assertEqual(row_indices[i], -1)
                self.assertEqual(row_values[i], 0)
            else:
                self.assertEqual(row_indices[i], indices[0])
                self.assertEqual(row_values[i], d1)

    def test_find_cross_check_and_ratio_test_matches(self):
        C = np.random.rand(100, 101).astype(np.float32)
        ratio = 0.7
        row_indices, row_values = mops.find_cross_check_and_ratio_test_matches(
            C, ratio=ratio
        )

        for i in range(100):
            indices = np.argsort(C[i])
            d1, d2 = C[i][indices][:2]

            # column minimum value
            cross_check_min_d1 = C[:, indices[0]].min()

            if d1 / d2 > ratio or cross_check_min_d1 != d1:
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

    def test_ratio_test_matchers(self):
        X = np.random.randint(0, 255, size=(105, 128)).astype(np.float32)
        Y = np.random.randint(0, 255, size=(100, 128)).astype(np.float32)

        Y[0, :] = X[10, :]

        ratio = 0.7
        fast_matcher = matchers.FastL2RTBFMatcher(ratio)
        cv_matcher = OpenCVL2RTBFMatcher(ratio)
        fast_result = fast_matcher.match(X, Y)

        cv_result = cv_matcher.match(X, Y)

        self.assertEqual(fast_result, cv_result)

    def test_matching_real_image(self):
        path = Path(os.path.dirname(os.path.abspath(__file__))).parent
        image = cv2.imread(str(path / "data/uber.jpg"))
        image = cv2.resize(image, (512, 512))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()

        _, des1 = sift.detectAndCompute(image, None)
        _, des2 = sift.detectAndCompute(image[:, ::-1], None)

        fast_matcher_rt = matchers.FastL2RTBFMatcher(ratio=0.7)
        fast_matcher_cc = matchers.FastL2CCBFMatcher()

        cv_matcher_rt = OpenCVL2RTBFMatcher(0.7)
        cv_matcher_cc = OpenCVL2CCBFMatcher()

        fs_match = fast_matcher_rt.match(des1, des2)
        cv_match = cv_matcher_rt.match(des1, des2)
        self.assertEqual(fs_match, cv_match)

        fs_match = fast_matcher_cc.match(des1, des2)
        cv_match = cv_matcher_cc.match(des1, des2)
        self.assertEqual(fs_match, cv_match)
