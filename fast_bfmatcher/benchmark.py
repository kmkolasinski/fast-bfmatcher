import numpy as np

import fast_bfmatcher.matchers as matchers
from fast_bfmatcher.matching_ops import find_row_col_min_values
from fast_bfmatcher.utils import measuretime


def benchmark(name: str, method, steps: int = 200, warmup: int = 10):

    with measuretime(f"{name:20} warmup", num_steps=steps, log=False):
        for _ in range(warmup):
            method()

    with measuretime(f"{name:20} calls ", num_steps=steps):
        for _ in range(steps):
            method()


def run(steps: int = 100, warmup: int = 5, num_kpts: int = 1000, dim: int = 128):

    benchmark_tf = True
    try:
        tf_matcher = matchers.TFL2BFMatcher()
    except Exception as e:
        print(f"Skipping tensorflow benchmark, got error: {e}")
        benchmark_tf = False

    X = np.random.randint(0, 255, size=(num_kpts, dim)).astype(np.float32)
    Y = np.random.randint(0, 255, size=(num_kpts, dim)).astype(np.float32)
    D = np.random.randint(0, 255, size=(num_kpts, num_kpts)).astype(np.float32)

    fast_matcher = matchers.FastBFL2Matcher()
    cv_matcher = matchers.CVBFL2Matcher()
    np_matcher = matchers.NumpyBFL2Matcher()

    print("\n>> Benchmarking matchers ...")
    benchmark("fast", lambda: fast_matcher.match(X, Y), steps=steps, warmup=warmup)
    benchmark("opencv", lambda: cv_matcher.match(X, Y), steps=steps, warmup=warmup)
    benchmark("numpy", lambda: np_matcher.match(X, Y), steps=steps, warmup=warmup)

    if benchmark_tf:
        tf_matcher = matchers.TFL2BFMatcher()
        benchmark(
            "tensorflow", lambda: tf_matcher.match(X, Y), steps=steps, warmup=warmup
        )

    print("\n>> Benchmarking distance matrix computation ...")

    benchmark(
        "fast",
        lambda: matchers.FastBFL2Matcher.l2_distance_matrix(X, Y),
        steps=steps,
        warmup=warmup,
    )
    benchmark(
        "numpy",
        lambda: matchers.NumpyBFL2Matcher.l2_distance_matrix(X, Y),
        steps=steps,
        warmup=warmup,
    )

    if benchmark_tf:
        benchmark(
            "tensorflow",
            lambda: matchers.TFL2BFMatcher.l2_distance_matrix(X, Y),
            steps=steps,
            warmup=warmup,
        )

    print("\n>> Benchmarking find row col min values and indices ...")

    def _find_row_col_min_values_np(x):
        row_indices_np = x.argmin(1)
        row_values_np = x.min(1)
        col_values_np = x.min(0)
        return row_indices_np, row_values_np, col_values_np

    benchmark("cython", lambda: find_row_col_min_values(D), steps=steps, warmup=warmup)
    benchmark(
        "numpy", lambda: _find_row_col_min_values_np(D), steps=steps, warmup=warmup
    )

    if benchmark_tf:
        benchmark(
            "tensorflow",
            lambda: matchers.TFL2BFMatcher.find_row_col_min_values(D),
            steps=steps,
            warmup=warmup,
        )
