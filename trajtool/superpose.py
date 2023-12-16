from .procrustes import solve as solve_procrustes
from .tfile import TFile


def superpose(tf: TFile):
    universe = tf.universe
    trajectory = universe.trajectory

    for itraj, traj in enumerate(trajectory):
        pass
