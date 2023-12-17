from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from .pbc import PBC
from .tfile import Index0


def radii(coords: NDArray, atoms: Index0.Group, centers: NDArray) -> NDArray:
    c = np.zeros(len(coords))
    dr = coords - centers
    rs = np.linalg.norm(dr, axis=-1)
    for lst in atoms:
        c[lst] = np.max(rs[lst])
    return c


def geom_centers(coords: NDArray, atoms: Index0.Group, box: PBC) -> NDArray:
    r0 = np.zeros_like(coords)
    for lst in atoms:
        r0[lst] = coords[lst[0]]
    rs = coords - r0
    rs = box.image(rs)
    c0 = np.zeros_like(coords)
    for lst in atoms:
        c0[lst] = np.sum(rs[lst], axis=0) / len(lst)
    return c0 + r0


def geom_center(coords: NDArray, atoms: NDArray, box: PBC) -> NDArray:
    r0 = coords[atoms[0]]
    rs = coords[atoms] - r0
    c0 = np.sum(box.image(rs), axis=0)
    return r0 + c0 / len(atoms)


def orthogonal_procrustes(src: NDArray, ref: NDArray) -> Tuple[NDArray, float]:
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
