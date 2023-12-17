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

    universe_ref = tf.universe_ref
    box_ref = PBC.from6(universe_ref.dimensions)
    crds_ref = universe_ref.coord.positions
    b_ct_ref = Vectrz.geom_center(crds_ref, b_indx, box_ref)
    b_crds_ref = crds_ref[b_indx] - b_ct_ref

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
        rctij = box.image(rctij)
        rctij = np.cumsum(rctij, axis=0)

        r_crds = box.image(coords - r_centers)
        for i, lst in enumerate(r_indx):
            r_crds[lst] += rctij[i]

        # move molecule centers in the box
        m_centers = Vectrz.geom_centers(r_crds, m_indx, box)
        m_crds = r_crds - m_centers + box.image(m_centers)

        # move backbone center to origin
        m_crds -= Vectrz.geom_center(m_crds, b_indx, box)

        # superpose
        b_crds = m_crds[b_indx]
        R, _rmsd = solve_procrustes(b_crds, b_crds_ref)

        # save coords
        tf.universe.trajectory[itraj].positions = m_crds.dot(np.transpose(R))
