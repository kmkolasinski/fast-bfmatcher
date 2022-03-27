import cv2
import numpy as np

from fast_bfmatcher.matchers import Matcher, MatchResult


class OpenCVL2CCBFMatcher(Matcher):
    """Reference openCV implementation"""

    def __init__(self):
        self.cv_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    def match(self, X: np.ndarray, Y: np.ndarray) -> MatchResult:
        X, Y = self.cast_inputs(X, Y)
        matches = self.cv_matcher.knnMatch(X, Y, k=1)
        matches_pairs = []
        distances = []
        for mm in matches:
            if len(mm) == 1:
                match = mm[0]
                row, col = match.queryIdx, match.trainIdx
                matches_pairs.append([row, col])
                distances.append(match.distance)

        return MatchResult(np.array(matches_pairs), np.array(distances))


class OpenCVL2RTBFMatcher(Matcher):
    def __init__(self, ratio: float = 0.7):
        self.ratio = ratio
        self.cv_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def match(self, X: np.ndarray, Y: np.ndarray) -> MatchResult:
        X, Y = self.cast_inputs(X, Y)
        matches = self.cv_matcher.knnMatch(X, Y, k=2)
        matches_pairs = []
        distances = []

        for i in range(len(matches)):
            match = matches[i][0]
            if match.distance <= self.ratio * matches[i][1].distance:
                row, col = match.queryIdx, match.trainIdx
                matches_pairs.append([row, col])
                distances.append(match.distance)

        return MatchResult(np.array(matches_pairs), np.array(distances))
