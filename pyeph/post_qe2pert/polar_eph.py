# long-range (dipole) polar part of the e-ph vertex
# following Perturbo's eph_wan_longrange_3d in polar_correction.f90
import numpy as np

def eph_wan_longrange_3d(polar_params, bg, tau, volume, tpiba, nat, qpoint):
    """
    Long-range vertex in cartesian atomic displacement coordinates, Rydberg
    units, atom-major packing (im = 3*ia + ix). Returns zeros for |q| < 1e-8
    (same Gamma cutoff as the Fortran).

    polar_params: dict from PhononDispersion.setup_polar_correction
    bg: reciprocal lattice vectors in 2pi/alat (3, 3)
    tau: atomic positions, cartesian, units of alat (nat, 3)
    """
    nmodes = 3 * nat
    epmatlr = np.zeros(nmodes, dtype=np.complex128)

    qpoint = np.asarray(qpoint, dtype=float)
    if np.sqrt(qpoint @ qpoint) < 1.0e-8:
        return epmatlr

    epsil = polar_params['epsil']
    zstar = polar_params['zstar']
    alpha = polar_params['polar_alpha']
    # nrx here, NOT nrx_ph (the epsilon-metric bounds)
    nrx1, nrx2, nrx3 = polar_params['nrx']

    falph = 4.0 * alpha
    ggmax = polar_params['gmax'] * falph

    for m1 in range(-nrx1, nrx1 + 1):
        for m2 in range(-nrx2, nrx2 + 1):
            for m3 in range(-nrx3, nrx3 + 1):
                xqr = bg.T @ (qpoint + np.array([m1, m2, m3]))  # cart, tpiba units
                qeq = xqr @ epsil @ xqr
                if qeq < 1.0e-14 or qeq > ggmax:
                    continue
                qfac = np.exp(-qeq / falph) / qeq
                for ia in range(nat):
                    arg = -2.0 * np.pi * (xqr @ tau[ia])
                    phase = np.cos(arg) + 1j * np.sin(arg)
                    # Fortran does matmul(xqr, zeu) with zeu = zstar[ia].T
                    # (h5 transpose), so this is zstar[ia] @ xqr
                    epmatlr[3 * ia:3 * ia + 3] += (zstar[ia] @ xqr) * qfac * phase

    # i * 4pi * e^2 / Omega / tpiba, e^2 = 2 in Ry units
    prefac = 1j * 4.0 * np.pi * 2.0 / volume / tpiba
    return prefac * epmatlr

def eph_wan_longrange(polar_params, bg, tau, volume, tpiba, nat, qpoint, uf=None):
    """
    Same but rotated to eigenmode coordinates when uf is given (mass-divided
    phonon eigenvectors, columns are modes): g_nu = sum_i uf[i, nu] ep[i],
    plain transpose, no conj.
    """
    ep = eph_wan_longrange_3d(polar_params, bg, tau, volume, tpiba, nat, qpoint)
    if uf is not None:
        ep = uf.T @ ep
    return ep
