import numpy as np


def solve(src: np.array, ref: np.array):
    """
    R = argmin_O |O.A - B| subject to Ot.O = 1.

    size of src: n x 3, aka At.
    size of ref: n x 3, aka Bt.

    |R.A - B| = |At.Rt - Bt|

    Returns R
    """
    # A = np.transpose(src)
    # B = np.transpose(ref)
    M = np.transpose(ref).dot(src)
    U, _S, Vt = np.linalg.svd(M)
    R = U.dot(Vt)
    et = src.dot(np.transpose(R)) - ref
    rmsd = np.sqrt(np.mean(et * et))
    return R, rmsd
