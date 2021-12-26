import numpy as np

import fast_bfmatcher.matchers as matchers
from fast_bfmatcher.utils import measuretime


def benchmark(name: str, method, steps: int = 100, warmup: int = 5):
    print()
    with measuretime(f"{name} warmup", num_steps=steps):
        for _ in range(warmup):
            method()

    with measuretime(f"{name} calls ", num_steps=steps):
        for _ in range(steps):
            method()


def run(steps: int = 100, warmup: int = 5, num_kpts: int = 1000, dim: int = 128):

    X = np.random.randint(0, 255, size=(num_kpts, dim)).astype(np.float32)
    Y = np.random.randint(0, 255, size=(num_kpts, dim)).astype(np.float32)

    fast_matcher = matchers.FastBFL2Matcher()
    cv_matcher = matchers.CVBFL2Matcher()
    np_matcher = matchers.NumpyBFL2Matcher()

    benchmark("fast", lambda: fast_matcher.match(X, Y), steps=steps, warmup=warmup)
    benchmark("opencv", lambda: cv_matcher.match(X, Y), steps=steps, warmup=warmup)
    benchmark("numpy", lambda: np_matcher.match(X, Y), steps=steps, warmup=warmup)
