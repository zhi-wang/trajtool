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
