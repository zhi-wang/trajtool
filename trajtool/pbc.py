from enum import IntEnum

import numpy as np
from numpy.typing import NDArray


class PBC:

    class Lattice(IntEnum):
        TRICLINIC = 1
        MONOCLINIC = 2
        ORTHOGONAL = 3

    def __init__(self):
        self.lattice: PBC.Lattice = PBC.Lattice.TRICLINIC
        self.pbc, self.rpbc = None, None

    @classmethod
    def from6(cls, dim6) -> "PBC":
        """
        Convert (a, b, c, al_deg, be_deg, ga_deg) to
        L = L = [[ax, bx, cx],
             [0., by, cy],
             [0., 0., cz]]
        and R = inv(L), such that F = R.C and C = L.F.
        """

        p = PBC()

        if dim6[3] == 90. and dim6[5] == 90.:
            p.lattice = PBC.Lattice.MONOCLINIC
            if dim6[4] == 90.:
                p.lattice = PBC.Lattice.ORTHOGONAL

        radian = 180. / np.pi
        a, b, c, al, be, ga = dim6
        al, be, ga = al / radian, be / radian, ga / radian

        if p.lattice == PBC.Lattice.ORTHOGONAL:
            ax, by, cz = a, b, c
            bx, cx, cy = 0., 0., 0.
        else:
            raise NotImplementedError(f"{p.lattice.name}")

        p.pbc = np.array([[ax, bx, cx], [0., by, cy], [0., 0., cz]])
        p.rpbc = np.linalg.inv(p.pbc)

        return p

    def f2c(self, frac: NDArray) -> NDArray:
        if frac.ndim == 2:
            return frac.dot(np.transpose(self.pbc))
        else:
            return self.pbc.dot(frac)

    def c2f(self, cart: NDArray) -> NDArray:
        if cart.ndim == 2:
            return cart.dot(np.transpose(self.rpbc))
        else:
            return self.rpbc.dot(cart)

    def image(self, r: NDArray) -> NDArray:
        f1 = self.c2f(r)
        f2 = f1 - np.floor(0.5 + f1)
        return self.f2c(f2)
