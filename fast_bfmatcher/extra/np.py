import numpy as np

from fast_bfmatcher.matchers import Matcher, MatchResult


class NumpyL2CCBFMatcher(Matcher):
    """Reference numpy implementation"""

    @classmethod
    def l2_distance_matrix(cls, X: np.ndarray, Y: np.ndarray):
        sqnorm1 = np.sum(np.square(X), 1, keepdims=True)
        sqnorm2 = np.sum(np.square(Y), 1, keepdims=True)
        innerprod = np.dot(X, Y.T)
        return sqnorm1 + np.transpose(sqnorm2) - 2.0 * innerprod

    def match(self, X: np.ndarray, Y: np.ndarray) -> MatchResult:

        dist_mat = self.l2_distance_matrix(X, Y)

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
