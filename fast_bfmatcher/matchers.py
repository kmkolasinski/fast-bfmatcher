import dataclasses
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from fast_bfmatcher import matching_ops as mops


@dataclasses.dataclass(frozen=True)
class MatchResult:
    indices: np.ndarray
    """
    Integer array of shape [num_matches, 2] with (i, j) pair
    matches indices between X[i, :] and Y[j, :] descriptors
    """
    distances: np.ndarray
    """
    Float array of shape [num_matches] with computed distances
    between matched (i, j) pairs.
    """

    def __eq__(self, other: "MatchResult"):
        try:
            np.testing.assert_equal(self.indices, other.indices)
            np.testing.assert_almost_equal(self.distances, other.distances, decimal=5)
        except AssertionError:
            return False
        return True


class Matcher(ABC):
    @abstractmethod
    def match(self, X: np.ndarray, Y: np.ndarray) -> MatchResult:
        pass

    @classmethod
    def cast_input(cls, X: np.ndarray, dtype: np.dtype) -> np.ndarray:
        if X.dtype != dtype:
            X = X.astype(dtype)
        return X

    @classmethod
    def cast_inputs(
        cls, X: np.ndarray, Y: np.ndarray, dtype: np.dtype = np.float32
    ) -> Tuple[np.ndarray, np.ndarray]:
        X = cls.cast_input(X, dtype)
        Y = cls.cast_input(Y, dtype)
        return X, Y


class FastL2CCBFMatcher(Matcher):
    """Brute force matcher equivalent to cv2.BFMatcher(NORM_L2, cross_check=True)"""

    @classmethod
    def l2_distance_matrix(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        D = np.zeros([X.shape[0], Y.shape[0]], np.float32)
        mops.l2_distance_matrix(X, Y, D)
        return D

    def match(self, X: np.ndarray, Y: np.ndarray) -> MatchResult:
        X, Y = self.cast_inputs(X, Y)
        indices, distances = mops.l2_cross_check_matcher(X, Y)
        return MatchResult(indices, distances)


class FastL2RTBFMatcher(Matcher):
    """
    Brute force matcher equivalent to cv2.BFMatcher(NORM_L2, cross_check=False)
    With Lowe's ratio test
    """

    def __init__(self, ratio: float = 0.7):
        self.ratio = ratio

    def match(self, X: np.ndarray, Y: np.ndarray) -> MatchResult:
        X, Y = self.cast_inputs(X, Y)
        indices, distances = mops.l2_ratio_test_matcher(X, Y, ratio=self.ratio)
        return MatchResult(indices, distances)
