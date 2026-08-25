#!/usr/bin/env python
# build svd_kq.h5 from a perturbo PREFIX_epr.h5 with pyeph (replaces the
# modified perturbo calc_mode='svd-elph'). Copy next to your run, edit the
# parameters, run (serial or mpirun).
from pyeph.post_qe2pert.eph_svd import EphSVD

EPR = "lif_epwan.h5"
KMESH = [5, 5, 5]        # uniform k = q mesh, Gamma first, z fastest
NSVD = 20                # None = full rank
PHFREQ_CUTOFF_MEV = 1.0
POLAR = True
OUT = "gkq/svd_kq.h5"    # directory must exist

ep = EphSVD(EPR, polar=POLAR)
ep.dump_svd_kq(OUT, KMESH, nsvd=NSVD, phfreq_cutoff_mev=PHFREQ_CUTOFF_MEV)
