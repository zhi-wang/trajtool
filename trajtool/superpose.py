import numpy as np

from .pbc import PBC
from .procrustes import solve as solve_procrustes
from .tfile import Index0, TFile
from .vectrz import Vectrz


def superpose(tf: TFile):
    trajectory = tf.universe.trajectory

    r_indx = tf.residule_atoms()
    r_head = Index0.head(r_indx)

    b_indx = tf.backbone_atoms()
    m_indx = tf.molecule_atoms()

    for itraj, traj in enumerate(trajectory):
        jtraj = itraj + 1
        if jtraj % 10 == 0:
            print(f"superpose: {jtraj:>8d}")

        coords = traj.positions
        box = PBC.from6(traj.dimensions)

        # residue centers
        r_centers = Vectrz.geom_centers(coords, r_indx, box)

        # synthesize molecules from residues
        rct_i = r_centers[r_head]
        rctij = np.copy(rct_i)
        rctij[0] = np.zeros(3)
        rctij[1:] -= rct_i[0:-1]
        rctij = Vectrz.image(rctij, box)
        rctij = np.cumsum(rctij, axis=0)

        r_crds = Vectrz.image(coords - r_centers, box)
        for i, lst in enumerate(r_indx):
            r_crds[lst] += rctij[i]

        # move molecule centers in the box
        m_centers = Vectrz.geom_centers(r_crds, m_indx, box)
        m_crds = r_crds - m_centers + Vectrz.image(m_centers, box)

        # move backbone center to origin
        m_crds -= Vectrz.geom_center(m_crds, b_indx, box)

        # save coords
        coords = m_crds
        tf.universe.trajectory[itraj].positions = coords
