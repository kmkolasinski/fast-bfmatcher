# fast-bfmatcher

## Installation
```bash
 pip install git+https://github.com/kmkolasinski/fast-bfmatcher
```

## Usage

Faster replacement for

```python
import cv2
import numpy as np

X = np.random.randint(0, 255, size=(1000, 128)).astype(np.float32)
Y = np.random.randint(0, 255, size=(1005, 128)).astype(np.float32)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.knnMatch(X, Y, k=1)

indices = []
distances = []
for match in matches:
    if len(match) == 1:
        match = match[0]
        row, col = match.queryIdx, match.trainIdx
        indices.append([row, col])
        distances.append(match.distance)
```

Usage:

```python

from fast_bfmatcher.matchers import FastBFL2Matcher

fast_matcher = FastBFL2Matcher()
result = fast_matcher.match(X, Y)

result.indices, result.distances
```

# Info

The speed of the matcher depends on the:
- installed blas library
- system and CPU 