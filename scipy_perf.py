import time
from unittest.mock import patch
import numpy as np
from scipy.spatial import distance

coords = np.random.random((100, 2))
weights = [1, 2]

itime = time.time()
for _ in range(50):
    distance.cdist(coords, coords, metric='sqeuclidean', w=weights)

print(f"Time: {time.time() - itime}")

with patch('scipy.spatial.distance._validate_vector') as mock_requests:
    mock_requests.side_effect = lambda x, *args, **kwargs: np.asarray(x)

    itime = time.time()
    for _ in range(50):
        distance.cdist(coords, coords, 'sqeuclidean', w=weights)

    print(f"Time: {time.time() - itime}")