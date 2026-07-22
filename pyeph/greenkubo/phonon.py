import numpy
from pyeph.greenkubo.mp_qmesh import get_mp_qmesh_info
from pyeph.utils.logger import logger

class ClassicPhononBath:
    """
    An approximation when the phonon is very local in unit cell.
    -> Phonon hopping between cells is negligible.
    """
    def __init__(self, ph_freq, temperature, gmat, distribution='Boltzmann'):
        """
        For now I also firstly make approximation that these classical phonons are purely local, as we did for the quantum phonons.
        TODO: later we will need to sample from k-space, and do ifft to get the real space's displacement.
        """
        assert distribution in ['Boltzmann', 'Wigner']
        self.distribution = distribution
        self.beta = 1.0 / temperature
        self.w = ph_freq.mean(axis=1) # TODO: this assumption to be removed later.
        self.nmodes = len(self.w)
        self.gmat = gmat
        
    def initialize_position_and_momentum(self, nx, ny, ntraj, rng):
        self.q0 = numpy.zeros((self.nmodes, ntraj, nx * ny))
        self.p0 = numpy.zeros((self.nmodes, ntraj, nx * ny))
        
        '''
        # Debugging purpose, to compare with the old code (can be removed later)
        # initialize the Holstein modes
        nwH = 8
        p_holstein = numpy.zeros((nwH, ntraj, nx * ny * 2))
        q_holstein = numpy.zeros((nwH, ntraj, nx * ny * 2))
        for imode, w_h in enumerate(self.w[:nwH//2]):
            rng = numpy.random.default_rng(1120 + imode)
            q_holstein[imode] = rng.normal(0, numpy.sqrt(1 / (self.beta * w_h**2)), (ntraj, nx * ny * 2))
            q_holstein[imode] = q_holstein[imode] * numpy.sqrt(w_h * 2)
            p_holstein[imode] = rng.normal(0, numpy.sqrt(1 / self.beta), (ntraj, nx * ny * 2))            
            p_holstein[imode] = p_holstein[imode] * numpy.sqrt(2 / w_h)
            
            # old convention: (2 * 162 mols)
            # now we want: (2 molecules in a cell, 162 cells)
        
            for i in range(0, nx * ny * 2, nx):
                cell_y_idx = i // (nx * 2)
                mol_idx_in_cell = (i // nx) % 2
                self.q0[4*mol_idx_in_cell+imode, :, cell_y_idx * nx:(cell_y_idx + 1) * nx] = q_holstein[imode, :, i:i+nx]
                self.p0[4*mol_idx_in_cell+imode, :, cell_y_idx * nx:(cell_y_idx + 1) * nx] = p_holstein[imode, :, i:i+nx]
        
        import h5py
        with h5py.File('q_holstein.h5', 'w') as f:
            f.create_dataset('q_holstein', data=self.q0[:nwH])
            f.create_dataset('p_holstein', data=self.p0[:nwH])
        
        # initialize the Peierls modes
        offset = nwH
        for imode in range(3):
            rng = numpy.random.default_rng(1120)
            w_p = self.w[offset]
            p_init = rng.normal(0, numpy.sqrt(1 / self.beta), (ntraj, nx * ny * 2))
            q_init = rng.normal(0, numpy.sqrt(1 / (self.beta * w_p**2)), (ntraj, nx * ny * 2))
            q_init = q_init * numpy.sqrt(w_p * 2)
            p_init = p_init * numpy.sqrt(2 / w_p)
            for i in range(0, nx * ny * 2, nx):
                cell_y_idx = i // (nx * 2)
                mode_idx = (i // nx) % 2
                self.q0[offset+3*mode_idx+imode][:, cell_y_idx * nx:(cell_y_idx + 1) * nx] = q_init[:, i:i+nx]
                self.p0[offset+3*mode_idx+imode][:, cell_y_idx * nx:(cell_y_idx + 1) * nx] = p_init[:, i:i+nx]
        self.qfield = self.q0.copy()
        return
        '''
        # TODO: we need to check the EOM for both cases. should be correct.
        ncells = nx * ny
        self.nx = nx
        self.ny = ny
        if self.distribution == 'Wigner':
            arg = self.beta * self.w / 2
            coth_arg = 1.0 / numpy.tanh(arg)
            sigma = numpy.sqrt(coth_arg)
            self.q0 = numpy.zeros((self.nmodes, ntraj, ncells))
            self.p0 = numpy.zeros((self.nmodes, ntraj, ncells))
            for imode in range(self.nmodes):
                self.q0[imode] = rng.normal(0, sigma[imode], (ntraj, ncells))
                self.p0[imode] = rng.normal(0, sigma[imode], (ntraj, ncells))
        elif self.distribution == 'Boltzmann':
            self.q0 = numpy.zeros((self.nmodes, ntraj, ncells))
            self.p0 = numpy.zeros((self.nmodes, ntraj, ncells))
            for imode in range(self.nmodes):
                self.p0[imode] = rng.normal(0, numpy.sqrt(1 / self.beta), (ntraj, ncells))
                self.q0[imode] = rng.normal(0, numpy.sqrt(1 / (self.beta * self.w[imode]**2)), (ntraj, ncells))
            self.q0 = numpy.einsum("utc,u->utc", self.q0, numpy.sqrt(self.w * 2))
            self.p0 = numpy.einsum("utc,u->utc", self.p0, numpy.sqrt(2 / self.w))
        self.qfield = self.q0.copy()
    
    def update_position(self, t):
        self.qfield = numpy.einsum("utc,u->utc", self.q0, numpy.cos(self.w * t))
        self.qfield += numpy.einsum("utc,u->utc", self.p0, numpy.sin(self.w * t))


class ClassicalPhononNonlocal(ClassicPhononBath):
    def __init__(self, ph_freq, temperature, gmat, distribution='Boltzmann', use_gauge_phase=False):
        super().__init__(ph_freq, temperature, gmat, distribution)
        self.w = None
        self.w_half = ph_freq # (nmodes, nqpts_half)
        self.nmodes = ph_freq.shape[0]
        self.use_gauge_phase = use_gauge_phase
    
    def initialize_position_and_momentum(self, nx, ny, ntraj, rng):
        self.nx = nx
        self.ny = ny
        
        self.rgrids = []
        for rx in range(self.nx):
            for ry in range(self.ny):
                self.rgrids.append([rx, ry])
        self.rgrids = numpy.array(self.rgrids)
        self.w_full, self.q_half, self.q_partner, self.half_ij, self.partner_ij, self.qgrid_full = get_mp_qmesh_info(nx, ny, self.w_half) # (nmodes, nx, ny)
        rdotq = numpy.einsum("ra,xya->rxy", self.rgrids, self.qgrid_full)
        self.expiqr = numpy.exp(1j * 2.0 * numpy.pi * rdotq) / numpy.sqrt(nx * ny)
        
        ## only for debugging purpose
        rng2 = numpy.random.default_rng(0)
        gauge_phase_half = rng2.uniform(0, 2 * numpy.pi, (self.nmodes, len(self.q_half))) # complex random phase with shape (nmodes, len(q_half))
        gauge_phase_half = numpy.exp(1j * gauge_phase_half)
        gauge_phase_full = numpy.zeros((self.nmodes, self.nx, self.ny), dtype=numpy.complex128)
        hi, hj = self.half_ij[:, 0], self.half_ij[:, 1]
        gauge_phase_full[:, hi, hj] = gauge_phase_half
        pi, pj = self.partner_ij[:, 0], self.partner_ij[:, 1]
        gauge_phase_full[:, pi, pj] = gauge_phase_half.conj()
        self.gauge_phase = gauge_phase_full
        
        self.ntraj = ntraj
        n_half_qpts = self.w_half.shape[1]
        q0_half_real = numpy.zeros((self.nmodes, ntraj, n_half_qpts))
        q0_half_imag = numpy.zeros((self.nmodes, ntraj, n_half_qpts))
        p0_half_real = numpy.zeros((self.nmodes, ntraj, n_half_qpts))
        p0_half_imag = numpy.zeros((self.nmodes, ntraj, n_half_qpts))
        
        if self.distribution == "Boltzmann":
            for imode in range(self.nmodes):
                for iq in range(n_half_qpts):
                    p0_half_real[imode, :, iq] = rng.normal(0, numpy.sqrt(1 / self.beta) / numpy.sqrt(2), (ntraj, ))
                    p0_half_imag[imode, :, iq] = rng.normal(0, numpy.sqrt(1 / self.beta) / numpy.sqrt(2), (ntraj, ))
                    q0_half_real[imode, :, iq] = rng.normal(0, numpy.sqrt(1 / (self.beta * self.w_half[imode, iq]**2)) / numpy.sqrt(2), (ntraj, ))
                    q0_half_imag[imode, :, iq] = rng.normal(0, numpy.sqrt(1 / (self.beta * self.w_half[imode, iq]**2)) / numpy.sqrt(2), (ntraj, ))
                
            q0_half = q0_half_real + 1j * q0_half_imag
            p0_half = p0_half_real + 1j * p0_half_imag
            
            self.q0_half = numpy.einsum("utq,uq->utq", q0_half, numpy.sqrt(self.w_half * 2))
            self.p0_half = numpy.einsum("utq,uq->utq", p0_half, numpy.sqrt(2 / self.w_half))
        elif self.distribution == "Wigner":
            for imode in range(self.nmodes):
                for iq in range(n_half_qpts):
                    arg = self.beta * self.w_half[imode, iq] / 2
                    coth_arg = 1.0 / numpy.tanh(arg)
                    sigma = numpy.sqrt(coth_arg) / numpy.sqrt(2)
                    q0_half_real[imode, :, iq] = rng.normal(0, sigma, (ntraj, ))
                    q0_half_imag[imode, :, iq] = rng.normal(0, sigma, (ntraj, ))
                    p0_half_real[imode, :, iq] = rng.normal(0, sigma, (ntraj, ))
                    p0_half_imag[imode, :, iq] = rng.normal(0, sigma, (ntraj, ))
                
            self.q0_half = q0_half_real + 1j * q0_half_imag
            self.p0_half = p0_half_real + 1j * p0_half_imag
        else:
            raise ValueError(f"Distribution {self.distribution} not supported")
        self.q0_full = self.reciprocal_half_to_full(self.q0_half)
        self.p0_full = self.reciprocal_half_to_full(self.p0_half)
        self.update_position(0)
        
    def update_position(self, t):
        qfield_reciprocal = numpy.einsum("utxy,uxy->utxy", self.q0_full, numpy.cos(self.w_full * t))
        qfield_reciprocal += numpy.einsum("utxy,uxy->utxy", self.p0_full, numpy.sin(self.w_full * t))
        self.qfield = self.position_reciprocal_to_real(qfield_reciprocal)
    
    def reciprocal_half_to_full(self, q_half):
        q_full = numpy.zeros((self.nmodes, self.ntraj, self.nx, self.ny), dtype=numpy.complex128)
        hi, hj = self.half_ij[:, 0], self.half_ij[:, 1]
        q_full[:, :, hi, hj] = q_half
        pi, pj = self.partner_ij[:, 0], self.partner_ij[:, 1]
        q_full[:, :, pi, pj] = q_half.conj()
        return q_full
    
    def position_reciprocal_to_real(self, q_reciprocal):
        """
        add back gauge:
        Xr(nu) = sum_q exp(iqr) Xq(nu) f(q,nu) / sqrt(Nx * Ny)
        without gauge:
        Xr(nu) = sum_q exp(iqr) Xq(nu) / sqrt(Nx * Ny)
        
        q_reciprocal : (nmodes, ntraj, nqx, nqy) ndarray, Complex128
        expiqr : (nrgrid, nqx, nqy) ndarray, Complex128
        gauge_phase : (nmodes, nqx, nqy) ndarray, Complex128
        """
        if self.use_gauge_phase:
            Xr = numpy.einsum('utxy, rxy, uxy->utr', q_reciprocal, self.expiqr, self.gauge_phase)
        else:
            Xr = numpy.einsum("utxy,rxy->utr", q_reciprocal, self.expiqr)
        assert numpy.allclose(Xr.imag, 0.0)
        Xr = Xr.real
        return Xr
        
        # Xq_fft = numpy.fft.ifftshift(q_reciprocal, axes=(-2, -1))
        # Xr = numpy.fft.ifft2(Xq_fft, axes=(-2, -1))
        # Xr *= numpy.sqrt(self.nx * self.ny) # yea, this should be commented.

        # # phase correction for your MP definition when N even
        # sx = 0.5 / self.nx if (self.nx % 2 == 0) else 0.0
        # sy = 0.5 / self.ny if (self.ny % 2 == 0) else 0.0
        # if sx != 0.0 or sy != 0.0:
        #     ix = numpy.arange(self.nx)[:, None]
        #     iy = numpy.arange(self.ny)[None, :]
        #     phase = numpy.exp(2j * numpy.pi * (sx * ix + sy * iy))
        #     Xr *= phase
        # Xr = Xr.real
        # return Xr.reshape(self.nmodes, self.ntraj, -1)
   
class QuantumPhononBath:
    def __init__(self, ph_freq, gmat, temperature, band_narrow_only=False):
        """
        ph_freq : (nmodes,) ndarray
            Phonon frequencies for each mode.
        gmat : (nmodes,) ndarray
            E-ph couplings for each mode.
        """
        self.w = ph_freq
        # check the validity of the phonon frequencies
        assert self.w.min() > 0, "phonon frequencies must be positive"
        valid_mask = numpy.abs(self.w) >= 1e-12
        self.w = self.w[valid_mask]
        gmat = gmat[valid_mask]
        
        self.g_dimless = gmat / self.w
        self.beta = 1.0 / temperature
        phi0 = (self.g_dimless **2 / numpy.tanh(self.beta * self.w / 2)).sum()
        self.phi0 = phi0
        self.phit = phi0
        self.polaron_prefactor = numpy.exp(-self.phi0)
        logger.info(f"Polaron transform prefactor: {self.polaron_prefactor}")
        self.exponents = numpy.array([-2, -1, 0, 1, 2])
        self.sector_weights = numpy.exp(-self.exponents * self.phi0)
        self.band_narrow_only = band_narrow_only
    
    def update_phit(self, t):
        phit = self.g_dimless **2 * (numpy.cos(self.w * t) / numpy.tanh(self.beta * self.w / 2) - 1.0j * numpy.sin(self.w * t))
        self.phit = phit.sum()
        self.sector_weights = numpy.exp(-self.exponents * self.phit)
        
def build_phonon_baths(ph_freq, gmat, cpa_cutoff, temperature, distribution, nonlocal_phonons=False, use_gauge_phase=False):
    """
    Split phonon modes into 'classical' and 'quantum' sets using a local (q-averaged) frequency,
    and extract local quantum e-ph couplings (onsite & diagonal in origin cell).

    Parameters
    ----------
    ph_freq : (nmodes, nqpts) ndarray
        Phonon frequencies for each mode and q-point from first-principles calculation.
    gmat : dict, nested dict: gmat[De][Dp] = (ncenter, ncenter, nmodes) array of e-ph couplings.
        De = (dx_e, dy_e) electron-cell displacement, Dp = (dx_p, dy_p) phonon-cell displacement.
    cpa_cutoff : float
        Threshold separating 'classical' (<= cutoff) from 'quantum' (> cutoff) modes.
    temperature : float
        Temperature
    seed : int
        Random seed.
    distribution : str
        Distribution of classical phonons.

    Returns
    -------
    """
    ph_freq_diag = ph_freq.mean(axis=1)
    classical_mask = ph_freq_diag <= cpa_cutoff
    ph_freq_classical = ph_freq[classical_mask, :]
    gmat_classical = {
        De: {Dp: epc_matrix[:, :, classical_mask].real for Dp, epc_matrix in inner.items()}
        for De, inner in gmat.items()
    }
    if nonlocal_phonons:
        classical_ph = ClassicalPhononNonlocal(ph_freq_classical, temperature, gmat_classical, distribution, use_gauge_phase=use_gauge_phase)
    else:
        classical_ph = ClassicPhononBath(ph_freq_classical, temperature, gmat_classical, distribution)
    ph_freq_quantum = ph_freq_diag[~classical_mask].ravel()
    
    if len(ph_freq_quantum) > 0 and len(ph_freq_classical) > 0:
        logger.info("Both classical and quantum modes are present, will build both baths.")
        logger.info(f"Cutoff frequency: {cpa_cutoff}")
        logger.info(f"Split {ph_freq.shape[0]} modes into {ph_freq_classical.shape[0]} classical and {ph_freq_quantum.shape[0]} quantum modes.")
    elif len(ph_freq_quantum) > 0:
        logger.info("Only quantum modes are present, will build quantum bath.")
    elif len(ph_freq_classical) > 0:
        logger.info("Only classical modes are present, will build classical bath.")
    else:
        raise ValueError("No phonon modes are present, check the input data.")
    
    if len(ph_freq_quantum) > 0:
        De = (0,0)
        Dp = (0,0)
        # we collect all the diagonal modes to a single list
        # TODO: here I make an approximation, that any |Dp| > 0 is negligible.
        # we should add a checker, or derive the p.t formalism when fast Frohlich-like modes are present.
        g_quantum = gmat[De][Dp][0, 0, ~classical_mask].real # (ncenter, ncenter, nmodes_quantum)
        quantum_ph = QuantumPhononBath(ph_freq_quantum, g_quantum, temperature)
        
        return classical_ph, quantum_ph
    
    return classical_ph, None
