import numpy as np
from typing import (
    TYPE_CHECKING,
    SupportsFloat,
    Union, Optional
)

if TYPE_CHECKING:
    import numpy.typing as npt

def mean(x: npt.ArrayLike,
         weights: Optional[Union[None, npt.ArrayLike]] = ...,
         axis: Optional[int] = ...,
         ddof: Optional[Union[int]] = ...,
         tol: Optional[Union[None, np.floating, SupportsFloat]] = ...) -> npt.ArrayLike:
    ...
