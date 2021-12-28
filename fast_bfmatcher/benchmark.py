import numpy as np

import fast_bfmatcher.matchers as matchers
from fast_bfmatcher.utils import measuretime


def benchmark(name: str, method, steps: int = 200, warmup: int = 10):

    with measuretime(f"{name:20} warmup", num_steps=steps, log=False):
        for _ in range(warmup):
            method()

    with measuretime(f"{name:20} calls ", num_steps=steps):
        for _ in range(steps):
            method()


def run(steps: int = 100, warmup: int = 5, num_kpts: int = 1000, dim: int = 128):

    X = np.random.randint(0, 255, size=(num_kpts, dim)).astype(np.float32)
    Y = np.random.randint(0, 255, size=(num_kpts, dim)).astype(np.float32)

    fast_matcher = matchers.FastBFL2Matcher()
    cv_matcher = matchers.CVBFL2Matcher()
    np_matcher = matchers.NumpyBFL2Matcher()

    print("\n>> Benchmarking matchers ...")
    benchmark("fast", lambda: fast_matcher.match(X, Y), steps=steps, warmup=warmup)
    benchmark("opencv", lambda: cv_matcher.match(X, Y), steps=steps, warmup=warmup)
    benchmark("numpy", lambda: np_matcher.match(X, Y), steps=steps, warmup=warmup)

    try:
        tf_matcher = matchers.TFL2BFMatcher()
        benchmark(
            "tensorflow", lambda: tf_matcher.match(X, Y), steps=steps, warmup=warmup
        )
    except Exception as e:
        print(f"Skipping tensorflow benchmark, got error: {e}")
        pass

    print("\n>> Benchmarking distance matrix computation ...")
    from fast_bfmatcher.matching_ops import l2_distance_matrix

    D = np.zeros([num_kpts, num_kpts], np.float32)
    benchmark("fast", lambda: l2_distance_matrix(X, Y, D))
    benchmark("numpy", lambda: matchers.NumpyBFL2Matcher.distance_matrix(X, Y))
