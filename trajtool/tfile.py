import argparse

import MDAnalysis as mda
import numpy as np


class TFile:

    def __init__(self, args: argparse.Namespace):
        self.universe = mda.Universe(*args.input)
        self.universe.transfer_to_memory()

        self.gsizes = np.array([int(a) for a in args.gsizes])
        self.msizes = np.array([int(a) for a in args.msizes])
        self.ngroups = len(self.gsizes)
        assert self.ngroups == len(self.msizes)

        self.nframes = len(self.universe.trajectory)
        self.natoms = self.universe.atoms.n_atoms
        assert sum(self.gsizes * self.msizes) == self.natoms

        if args.option == "superpose":
            self.refu = mda.Universe(args.input[0])
            self.refu.transfer_to_memory()

    def index0_molecule_atoms(self) -> tuple[list[list[int]], list[int]]:
        lst = []
        start = 0
        for na, nmol in zip(self.msizes, self.gsizes):
            for _imol in range(nmol):
                lst.append(list(range(start, start + na)))
                start += na
        pos = np.zeros(self.natoms)
        for i, m in enumerate(lst):
            pos[m] = i
        return lst, pos.tolist()

    def index0_residule_atoms(self) -> tuple[list[list[int], list[int]]]:
        lst = []
        for res in self.universe.residues:
            l2 = [a.index for a in res.atoms]
            l2.sort()
            lst.append(l2)
        pos = np.zeros(self.natoms)
        for i, m in enumerate(lst):
            pos[m] = i
        return lst, pos.tolist()
