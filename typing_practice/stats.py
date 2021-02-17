import numpy as np

def mean(x, weights=None, axis=-1, ddof=0, tol=None):
    x = np.asanyarray(x)
    assert ddof >= 0
    if x.size != 0:
        assert ddof < x.shape[axis]
    if weights is not None:
        weights = np.asanyarray(weights)
        assert np.all(weights >= 0)
        sow = np.sum(weights, axis=axis)
        eps = tol or np.finfo(weights.dtype).eps
        assert np.all(1 - eps <= sow <= 1 + eps)
    if weights is None:
        if x.size == 0:
            weights = np.array([], dtype=x.dtype)
        else:
            weights = 1/(x.shape[axis] - ddof) * np.ones_like(x)
    return np.sum(weights * x, axis=axis)
