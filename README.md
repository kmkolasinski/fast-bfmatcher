# fast-bfmatcher

Depends on https://github.com/flame/blis which will be downloaded during 
installation and compiled.

## Installation
```bash
 pip install git+https://github.com/kmkolasinski/fast-bfmatcher
 pip install fast-bfmatcher
```

## Quick command to check speedup

* CC stands for Cross-Check
* RT stands for ratio test i.e. Lowe's ratio test proposed in the original SIFT paper

```python
import os

os.environ["BLIS_NUM_THREADS"] = "4"

from fast_bfmatcher.benchmark import benchmark_cc_matchers

benchmark_cc_matchers()
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

from fast_bfmatcher.matchers import FastL2CCBFMatcher, FastL2RTBFMatcher

fast_matcher = FastL2CCBFMatcher()
result = fast_matcher.match(X, Y)

fast_matcher = FastL2RTBFMatcher(ratio=0.7)
result = fast_matcher.match(X, Y)

result.indices, result.distances
```

# Info

The speed of the matcher depends on the:
- installed blas library
- system and CPU 



# Building 

```bash
python setup.py build_ext --inplace
```

# Testing 

```bash
export BLIS_NUM_THREADS=8;
export OMP_NUM_THREADS=8
pytest
```