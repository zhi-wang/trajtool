from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def solve(src: NDArray, ref: NDArray) -> Tuple[NDArray, float]:
    """
    R = argmin_O |O.A - B| subject to Ot.O = 1.

    size of src: n x 3, aka At.
    size of ref: n x 3, aka Bt.

    |R.A - B| = |At.Rt - Bt|

    Returns R
    """
    # A = np.transpose(src); B = np.transpose(ref)
    # M = B.At = U.S.Vt;     R = U.Vt
    M = np.transpose(ref).dot(src)
    U, _S, Vt = np.linalg.svd(M)
    R = U.dot(Vt)
    et = src.dot(np.transpose(R)) - ref
    rmsd = np.sqrt(np.mean(et * et))
    return R, rmsd
