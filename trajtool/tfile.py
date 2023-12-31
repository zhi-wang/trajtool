import argparse

import MDAnalysis as mda
import numpy as np
from numpy.typing import NDArray


class Index0:

    Group = list[NDArray]

    @classmethod
    def head(cls, group: Group) -> NDArray:
        return np.array([lst[0] for lst in group])


class TFile:

    def __init__(self, args: argparse.Namespace):
        self.universe = mda.Universe(*args.input, in_memory=True)

        self.gsizes = np.array([int(a) for a in args.gsizes])
        self.msizes = np.array([int(a) for a in args.msizes])
        self.ngroups = len(self.gsizes)
        assert self.ngroups == len(self.msizes)

        self.nframes = len(self.universe.trajectory)
        self.natoms = self.universe.atoms.n_atoms
        assert sum(self.gsizes * self.msizes) == self.natoms

        if hasattr(args, "out"):
            self.out = args.out

        if args.option == "superpose":
            self.universe_ref = mda.Universe(args.input[0], in_memory=True)

    def write_traj(self):
        ag = self.universe.select_atoms(f"id 1:{self.natoms}")
        ag.write(self.out, frames=self.universe.trajectory[:])

    def molecule_atoms(self) -> Index0.Group:
        lst = []
        start = 0
        for na, nmol in zip(self.msizes, self.gsizes):
            for _imol in range(nmol):
                lst.append(np.array(range(start, start + na)))
                start += na
        return lst

    def residule_atoms(self) -> Index0.Group:
        lst = []
        for res in self.universe.residues:
            l2 = [a.index for a in res.atoms]
            l2.sort()
            lst.append(np.array(l2))
        return lst

    def backbone_atoms(self) -> NDArray:
        return np.array([int(a.index) for a in self.universe.select_atoms("backbone")])
