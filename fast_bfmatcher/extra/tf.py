import numpy as np
import tensorflow as tf

from fast_bfmatcher.matchers import Matcher, MatchResult

if tf.__version__ < "2.0.0":
    if tf.__version__ >= "1.14.0":
        if not tf.compat.v1.executing_eagerly():
            msg = "TFL2BFMatcher requires eager mode to be enabled! Call tf.enable_eager_execution() first."
            print(msg)
            raise ImportError(msg)
    else:
        msg = f"TF Matcher Requires minimum version of Tensorflow is 2.0.0, got: {tf.__version__}"
        print(msg)
        raise ImportError(msg)


@tf.function(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.float32),))
def find_row_col_min_values_tf(X: tf.Tensor):
    row_indices = tf.argmin(X, 1)
    row_values = tf.reduce_min(X, 1)
    col_values = tf.reduce_min(X, 0)
    return row_indices, row_values, col_values


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    )
)
def l2_distance_matrix_tf(X: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
    sqnorm1 = tf.reduce_sum(tf.square(X), 1, keepdims=True)
    sqnorm2 = tf.reduce_sum(tf.square(Y), 1, keepdims=True)
    innerprod = tf.matmul(X, Y, transpose_a=False, transpose_b=True)
    return sqnorm1 + tf.transpose(sqnorm2) - 2 * innerprod


class TFL2CCBFMatcher(tf.Module, Matcher):
    def __init__(self, name: str = None):
        super(TFL2CCBFMatcher, self).__init__(name=name)
        self.dtype = tf.float32

    @classmethod
    def cast_input(cls, X: np.ndarray, dtype: np.dtype) -> np.ndarray:
        if X.dtype != dtype:
            X = tf.cast(X, dtype)
        return X

    @classmethod
    def l2_distance_matrix(cls, X: np.ndarray, Y: np.ndarray):
        D = l2_distance_matrix_tf(X, Y)
        return D.numpy()

    @classmethod
    def find_row_col_min_values(cls, X: np.ndarray):
        row_indices, row_values, col_values = find_row_col_min_values_tf(X)
        return row_indices.numpy(), row_values.numpy(), col_values.numpy()

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None], dtype=tf.float32),
        )
    )
    def match_fn(self, X: tf.Tensor, Y: tf.Tensor):

        dist_mat = l2_distance_matrix_tf(X, Y)

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


class TFL2RTBFMatcher(TFL2CCBFMatcher):
    def __init__(self, ratio: float = 0.7, name: str = None):
        super(TFL2CCBFMatcher, self).__init__(name)
        self.ratio = ratio

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None], dtype=tf.float32),
        )
    )
    def match_fn(self, X: tf.Tensor, Y: tf.Tensor):

        dist_mat = l2_distance_matrix_tf(X, Y)

        neg_distances, indices = tf.nn.top_k(-dist_mat, 2)
        distances = -neg_distances
        ratio2 = self.ratio ** 2
        is_good = distances[:, 0] < ratio2 * distances[:, 1]

        num_rows = tf.shape(dist_mat)[0]
        row_indices = tf.range(0, num_rows, dtype=indices.dtype)

        rows = row_indices[is_good]
        cols = indices[is_good][:, 0]
        distances = distances[is_good][:, 0]

        indices = tf.transpose(tf.stack([rows, cols]))
        distances = tf.sqrt(tf.maximum(0.0, distances))
        return indices, distances

    def match(self, X: np.ndarray, Y: np.ndarray) -> MatchResult:
        X, Y = self.cast_inputs(X, Y)
        indices, distances = self.match_fn(X, Y)
        return MatchResult(indices.numpy(), distances.numpy())
