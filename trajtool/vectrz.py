import numpy as np

from .pbc import PBC
from .tfile import Index0


class Vectrz:

    @classmethod
    def f2c(cls, frac: np.array, L: np.array) -> np.array:
        return frac.dot(np.transpose(L))

    @classmethod
    def c2f(cls, cart: np.array, R: np.array) -> np.array:
        return cart.dot(np.transpose(R))

    @classmethod
    def image(cls, r: np.array, box: PBC) -> np.array:
        f1 = cls.c2f(r, box.rpbc)
        f2 = f1 - np.floor(0.5 + f1)
        return cls.f2c(f2, box.pbc)

    @classmethod
    def radii(cls, coords: np.array, atoms: Index0.Group, centers: np.array) -> np.array:
        c = np.zeros(len(coords))
        dr = coords - centers
        rs = np.linalg.norm(dr, axis=-1)
        for lst in atoms:
            c[lst] = np.max(rs[lst])
        return c

    @classmethod
    def geom_centers(cls, coords: np.array, atoms: Index0.Group, box: PBC) -> np.array:
        c = np.zeros_like(coords)
        for lst in atoms:
            na = len(lst)
            r0 = coords[lst[0]]
            rs = coords[lst] - r0
            c0 = np.sum(cls.image(rs, box), axis=0)
            c[lst] = r0 + c0 / na
        return c

    @classmethod
    def geom_center(cls, coords: np.array, atoms: np.array, box: PBC) -> np.array:
        na = len(atoms)
        r0 = coords[atoms[0]]
        rs = coords[atoms] - r0
        c0 = np.sum(cls.image(rs, box), axis=0)
        return r0 + c0 / na
