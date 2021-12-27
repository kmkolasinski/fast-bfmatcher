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
            X = tf.cast(X, dtype)
        return X

    @classmethod
    def cast_inputs(
        cls, X: np.ndarray, Y: np.ndarray, dtype: np.dtype = np.float32
    ) -> Tuple[np.ndarray, np.ndarray]:
        X = cls.cast_input(X, dtype)
        Y = cls.cast_input(Y, dtype)
        return X, Y


class FastBFL2Matcher(Matcher):
    """Brute force matcher equivalent to cv2.BFMatcher(NORM_L2, cross_check=True)"""

    def match(self, X: np.ndarray, Y: np.ndarray) -> MatchResult:
        X, Y = self.cast_inputs(X, Y)
        indices, distances = mops.l2_cross_check_matcher(X, Y)
        return MatchResult(indices, distances)


class NumpyBFL2Matcher(Matcher):
    """Reference numpy implementation"""

    @classmethod
    def distance_matrix(cls, X: np.ndarray, Y: np.ndarray):
        sqnorm1 = np.sum(np.square(X), 1, keepdims=True)
        sqnorm2 = np.sum(np.square(Y), 1, keepdims=True)
        innerprod = np.dot(X, Y.T)
        return sqnorm1 + np.transpose(sqnorm2) - 2.0 * innerprod

    def match(self, X: np.ndarray, Y: np.ndarray) -> MatchResult:

        dist_mat = self.distance_matrix(X, Y)

        row_matches = np.argmin(dist_mat, 1)
        col_matches = np.argmin(dist_mat, 0)

        num_rows = row_matches.shape[0]

        inverse_row_indices = col_matches[row_matches]
        row_indices = np.arange(0, num_rows, dtype=row_matches.dtype)

        cross_checked = row_indices == inverse_row_indices
        rows = row_indices[cross_checked]
        cols = row_matches[cross_checked]

        indices = np.transpose(np.stack([rows, cols]))
        distances = dist_mat[rows, cols]
        distances = np.sqrt(np.maximum(0, distances))

        return MatchResult(indices, distances)


try:

    class CVBFL2Matcher(Matcher):
        """Reference openCV implementation"""

        def __init__(self):
            import cv2

            self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        def match(self, X: np.ndarray, Y: np.ndarray) -> MatchResult:
            X, Y = self.cast_inputs(X, Y)
            matches = self.bf.knnMatch(X, Y, k=1)
            matches_pairs = []
            distances = []
            for mm in matches:
                if len(mm) == 1:
                    match = mm[0]
                    row, col = match.queryIdx, match.trainIdx
                    matches_pairs.append([row, col])
                    distances.append(match.distance)

            return MatchResult(np.array(matches_pairs), np.array(distances))


except ImportError:
    pass


try:
    import tensorflow as tf

    if tf.__version__ < "2.0.0":
        if tf.__version__ >= "1.14.0":
            if not tf.compat.v1.executing_eagerly():
                msg = "TFL2BFMatcher requires eager mode to be enabled! Call tf.enable_eager_execution() first."
                print(msg)
                raise ImportError(msg)
        else:
            msg = f"TFL2BFMatcher Requires minimum version of Tensorflow is 2.0.0, got: {tf.__version__}"
            print(msg)
            raise ImportError(msg)

    class TFL2BFMatcher(tf.Module, Matcher):
        def __init__(self, name: str = None):
            super(TFL2BFMatcher, self).__init__(name=name)
            self.dtype = tf.float32

        @classmethod
        def distance_matrix(cls, X: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
            sqnorm1 = tf.reduce_sum(tf.square(X), 1, keepdims=True)
            sqnorm2 = tf.reduce_sum(tf.square(Y), 1, keepdims=True)
            innerprod = tf.matmul(X, Y, transpose_a=False, transpose_b=True)
            return sqnorm1 + tf.transpose(sqnorm2) - 2 * innerprod

        @tf.function(
            input_signature=(
                tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                tf.TensorSpec(shape=[None, None], dtype=tf.float32),
            )
        )
        def match_fn(self, X: tf.Tensor, Y: tf.Tensor):

            dist_mat = self.distance_matrix(X, Y)

            row_matches = tf.argmin(dist_mat, 1)
            col_matches = tf.argmin(dist_mat, 0)

            num_rows = tf.cast(tf.shape(row_matches)[0], dtype=row_matches.dtype)

            inverse_row_indices = tf.gather(col_matches, row_matches)
            row_indices = tf.range(0, num_rows, dtype=row_matches.dtype)

            cross_checked = tf.equal(row_indices, inverse_row_indices)
            rows = row_indices[cross_checked]
            cols = row_matches[cross_checked]

            indices = tf.transpose(tf.stack([rows, cols]))
            distances = tf.gather_nd(dist_mat, indices)
            distances = tf.sqrt(tf.maximum(0.0, distances))
            return indices, distances

        def match(self, X: np.ndarray, Y: np.ndarray) -> MatchResult:
            X, Y = self.cast_inputs(X, Y)
            indices, distances = self.match_fn(X, Y)
            return MatchResult(indices.numpy(), distances.numpy())


except ImportError:
    pass
