from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

import fast_bfmatcher.matchers as matchers
from fast_bfmatcher.extra.cv import OpenCVL2CCBFMatcher, OpenCVL2RTBFMatcher
from fast_bfmatcher.matching_ops import find_cross_check_matches
from fast_bfmatcher.utils import measuretime


def benchmark(name: str, method, steps: int = 200, warmup: int = 10):

    with measuretime(f"{name:20} warmup", num_steps=steps, log=False):
        for _ in range(warmup):
            method()

    with measuretime(f"{name:20} calls ", num_steps=steps):
        for _ in range(steps):
            method()


def benchmark_cc_matchers(
    steps: int = 100, warmup: int = 5, num_kpts: int = 2000, dim: int = 128
):

    benchmark_tf = True
    try:
        from fast_bfmatcher.extra.tf import TFL2CCBFMatcher

        tf_matcher = TFL2CCBFMatcher()
    except Exception as e:
        print(f"Skipping tensorflow benchmark, got error: {e}")
        benchmark_tf = False

    X = np.random.randint(0, 255, size=(num_kpts, dim)).astype(np.float32)
    Y = np.random.randint(0, 255, size=(num_kpts, dim)).astype(np.float32)
    D = np.random.randint(0, 255, size=(num_kpts, num_kpts)).astype(np.float32)

    fast_matcher = matchers.FastL2CCBFMatcher()

    cv_matcher = OpenCVL2CCBFMatcher()

    from fast_bfmatcher.extra.np import NumpyL2CCBFMatcher

    np_matcher = NumpyL2CCBFMatcher()

    print("\n>> Benchmarking matchers ...")
    benchmark("fast", lambda: fast_matcher.match(X, Y), steps=steps, warmup=warmup)
    benchmark("opencv", lambda: cv_matcher.match(X, Y), steps=steps, warmup=warmup)
    benchmark("numpy", lambda: np_matcher.match(X, Y), steps=steps, warmup=warmup)

    if benchmark_tf:
        benchmark(
            "tensorflow", lambda: tf_matcher.match(X, Y), steps=steps, warmup=warmup
        )

    print("\n>> Benchmarking distance matrix computation ...")

    benchmark(
        "fast", lambda: matchers.l2_distance_matrix(X, Y), steps=steps, warmup=warmup
    )
    benchmark(
        "numpy",
        lambda: NumpyL2CCBFMatcher.l2_distance_matrix(X, Y),
        steps=steps,
        warmup=warmup,
    )

    if benchmark_tf:
        benchmark(
            "tensorflow",
            lambda: TFL2CCBFMatcher.l2_distance_matrix(X, Y),
            steps=steps,
            warmup=warmup,
        )

    print("\n>> Benchmarking find row col min values and indices ...")

    def _find_row_col_min_values_np(x):
        row_indices_np = x.argmin(1)
        row_values_np = x.min(1)
        col_values_np = x.min(0)
        return row_indices_np, row_values_np, col_values_np

    benchmark("fast", lambda: find_cross_check_matches(D), steps=steps, warmup=warmup)
    benchmark(
        "numpy", lambda: _find_row_col_min_values_np(D), steps=steps, warmup=warmup
    )

    if benchmark_tf:
        benchmark(
            "tensorflow",
            lambda: TFL2CCBFMatcher.find_row_col_min_values(D),
            steps=steps,
            warmup=warmup,
        )


def benchmark_cc_rt_size_scan(
    max_size: int = 10000, random_sampler_per_size: int = 20, step: int = 500
) -> Dict[str, List[Any]]:
    """
    Measures matching time between two random int vectors of size [size, 128] in
    function of size parameter.

    NOTE: Before run set number of threads

            os.environ["BLIS_NUM_THREADS"] = "8"
            os.environ["OMP_NUM_THREADS"] = "8"
            cv2.setNumThreads(8)

    Args:
        max_size: maximum size of the tested vector
        random_sampler_per_size: number of random sampler per each size value
        step: scan step size

    Returns:
        metrics: dict with benchmark metrics
    """

    def _test_match(matcher: matchers.Matcher, name: str) -> Dict[str, List[Any]]:

        metrics = defaultdict(list)
        for n in range(100, max_size, step):
            X = np.random.randint(0, 255, size=(n, 128))
            Y = np.random.randint(0, 255, size=(n, 128))
            # warmup
            for _ in range(5):
                matcher.match(X, Y)

            with measuretime(f"{name} N={n}") as dt:
                for _ in range(random_sampler_per_size):
                    matcher.match(X, Y)

            metrics[name].append(1000 * dt.seconds / random_sampler_per_size)
            metrics["size"].append(n)

        return metrics

    fast_matcher_rt = matchers.FastL2RTBFMatcher(ratio=0.7)
    fast_matcher_cc = matchers.FastL2CCBFMatcher()
    fast_matcher_rt_cc = matchers.FastL2RTCCBFMatcher(ratio=0.7)

    cv_matcher_rt = OpenCVL2RTBFMatcher(0.7)
    cv_matcher_cc = OpenCVL2CCBFMatcher()

    fs_rt_metrics = _test_match(fast_matcher_rt, "fast-rt")
    fs_cc_metrics = _test_match(fast_matcher_cc, "fast-cc")
    fs_rt_cc_metrics = _test_match(fast_matcher_rt_cc, "fast-rt-cc")
    cv_rt_metrics = _test_match(cv_matcher_rt, "opencv-rt")
    cv_cc_metrics = _test_match(cv_matcher_cc, "opencv-cc")

    return {
        **fs_rt_metrics,
        **fs_cc_metrics,
        **fs_rt_cc_metrics,
        **cv_rt_metrics,
        **cv_cc_metrics,
    }
