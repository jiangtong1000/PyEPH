import numpy as np
from itertools import product

def parse_qpoint_path(qpoint_string):
    """Parse q-point path string and generate interpolated path"""
    lines = qpoint_string.strip().split('\n')
    nqpoints = int(lines[0])
    assert len(lines) == nqpoints + 1

    high_sym_points = []
    ninterp_list = []
        
    for i in range(1, nqpoints + 1):
        parts = lines[i].split()
        qx, qy, qz = float(parts[0]), float(parts[1]), float(parts[2])
        ninterp = int(parts[3])
        high_sym_points.append([qx, qy, qz]) # (nqpoints, 3)
        ninterp_list.append(ninterp) # (nqpoints,)

    qpath = []
    qpath_labels = []
        
    for i in range(len(high_sym_points) - 1):
        q1 = np.array(high_sym_points[i])
        q2 = np.array(high_sym_points[i + 1])
        
        # qpath.append(q1)
        ninterp = ninterp_list[i] + 1
        for j in range(ninterp):
            t = j / ninterp if ninterp > 1 else 0
            q_interp = q1 + t * (q2 - q1)
            qpath.append(q_interp)

    qpath.append(q2)
    qpath = np.array(qpath)
    return qpath

def get_length(r_cryst, at):
    r_cryst = np.asarray(r_cryst, dtype=np.float64)
    at = np.asarray(at, dtype=np.float64)
    r_cart = r_cryst @ at
    return np.linalg.norm(r_cart, axis=-1)

def set_cutoff_small(rdim, at):
    """
    Retrun the cutoff radius in real space for Wigner-Seitz cell vector search
    (the edge of the first Brillouin zone in reciprocal space)
    Define a sphere in real space containing all R-vectors within approx half of the Brillouin zone
    This ensures we capture all relevant R-vectors for Wannier hopping while avoiding unnecessary computations
    for R-vectors far away from the Brillouin zone.
    rdim: k-mesh dimensions [nk1, nk2, nk3]
    at: lattice vectors (3,3)
    """
    ndim = np.array(rdim) // 2 + 1 # half width of the k-mesh plus 1
    cutoff = 0.0
    
    for i, j, k in product([-1, 1], repeat=3): # corner points of the k-mesh
        r_cryst = ndim * np.array([i, j, k], dtype=np.float64)
        dist = get_length(r_cryst, at)
        if dist > cutoff:
            cutoff = dist
    
    return cutoff