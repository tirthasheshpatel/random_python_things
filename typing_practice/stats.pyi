import numpy as np
from typing import (
    TYPE_CHECKING,
    SupportsFloat,
    SupportsInt,
    Union, Optional
)

if TYPE_CHECKING:
    import numpy.typing as npt

def mean(x: npt.ArrayLike,
         weights: Optional[Union[None, npt.ArrayLike]] = ...,
         axis: Optional[SupportsInt] = ...,
         ddof: Optional[SupportsInt] = ...,
         tol: Optional[Union[None, SupportsFloat]] = ...) -> npt.ArrayLike:
    ...
