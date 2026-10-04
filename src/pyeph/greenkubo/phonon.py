"""Historical bath preparation with explicit native canonical coordinate maps.

Sampling order and half-grid conventions follow jiangtong1000/PyEPH revision
6c4693acbb69a06a5bc8b0593abde2170ff38843 (BSD-3-Clause). The simulation uses
the native harmonic schedule; these mutable fields are compatibility views.
"""

import numpy as np

from .mp_qmesh import get_mp_qmesh_info


def _temperature(temperature):
    if np.ndim(temperature) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive kB*T in the chosen energy unit")
    return float(temperature)


class ClassicPhononBath:
    nonlocal_phonons = False

    def __init__(self, ph_freq, temperature, gmat, distribution="Boltzmann"):
        if distribution not in {"Boltzmann", "Wigner"}:
            raise ValueError("distribution must be Boltzmann or Wigner")
        frequencies = np.asarray(ph_freq, dtype=float)
        if frequencies.ndim != 2 or frequencies.shape[1] < 1:
            raise ValueError("ph_freq must have shape (nmodes,nqpoints)")
        if not np.isfinite(frequencies).all() or np.any(frequencies <= 0):
            raise ValueError("sampled oscillator frequencies must be finite and positive")
        self.ph_freq = frequencies.copy()
        self.w = frequencies.mean(axis=1)
        self.beta = 1/_temperature(temperature)
        self.nmodes, self.gmat, self.distribution = len(self.w), gmat, distribution
        self._supplied_samples = None

    def set_initial_samples(self, q0, p0):
        """Use explicit legacy X,Y arrays instead of inferring seed equivalence.

        Local arrays are (mode,trajectory,cell); nonlocal arrays are complex
        half-grid amplitudes (mode,trajectory,qhalf). Inputs are copied.
        """
        q0, p0 = np.asarray(q0), np.asarray(p0)
        if q0.shape != p0.shape or q0.ndim != 3 or q0.shape[0] != self.nmodes:
            raise ValueError("initial samples require matching (mode,trajectory,cell_or_qhalf) arrays")
        if not np.isfinite(q0).all() or not np.isfinite(p0).all():
            raise ValueError("initial samples must be finite")
        if not self.nonlocal_phonons and (np.iscomplexobj(q0) or np.iscomplexobj(p0)):
            raise ValueError("local initial oscillator fields must be real")
        self._supplied_samples = (q0.copy(), p0.copy())
        return self

    def initialize_position_and_momentum(self, nx, ny, ntraj, rng):
        self.nx, self.ny, self.ntraj = int(nx), int(ny), int(ntraj)
        shape = self.nmodes, ntraj, nx*ny
        if self._supplied_samples is not None:
            if self._supplied_samples[0].shape != shape:
                raise ValueError(f"explicit local samples must have shape {shape}")
            self.q0, self.p0 = (x.copy() for x in self._supplied_samples)
        else:
            self.q0, self.p0 = np.zeros(shape), np.zeros(shape)
            for mode, w in enumerate(self.w):
                if self.distribution == "Wigner":
                    sigma = np.sqrt(1/np.tanh(self.beta*w/2))
                    self.q0[mode] = rng.normal(0, sigma, (ntraj, nx*ny))
                    self.p0[mode] = rng.normal(0, sigma, (ntraj, nx*ny))
                else:
                    # Preserve original p-then-q NumPy draw order exactly.
                    self.p0[mode] = rng.normal(0, np.sqrt(1/self.beta), (ntraj, nx*ny))
                    self.q0[mode] = rng.normal(0, np.sqrt(1/(self.beta*w**2)), (ntraj, nx*ny))
            if self.distribution == "Boltzmann":
                self.q0 = np.einsum("utc,u->utc", self.q0, np.sqrt(2*self.w))
                self.p0 = np.einsum("utc,u->utc", self.p0, np.sqrt(2/self.w))
        self.qfield = self.q0.copy()

    def update_position(self, t):
        self.qfield = (np.einsum("utc,u->utc", self.q0, np.cos(self.w*t))
                       + np.einsum("utc,u->utc", self.p0, np.sin(self.w*t)))

    def canonical_initial(self):
        if self.nmodes == 0:
            return np.zeros((self.ntraj, 1)), np.zeros((self.ntraj, 1))
        q = self.q0 / np.sqrt(2*self.w[:, None, None])
        p = self.p0 * np.sqrt(self.w[:, None, None]/2)
        return (q.transpose(1, 0, 2).reshape(self.ntraj, -1),
                p.transpose(1, 0, 2).reshape(self.ntraj, -1))

    def native_map(self):
        return {"frequencies": self.w,
                "canonical_frequencies": np.repeat(self.w, self.nx*self.ny)
                if self.nmodes else np.zeros(1)}

    def initial_samples(self):
        return self.q0.copy(), self.p0.copy()


class ClassicalPhononNonlocal(ClassicPhononBath):
    """Even MP half-grid modes with explicit complex-conjugate partners.

    Real-space output preserves the original x-major cell order. The native
    map records this ordering rather than assuming a conventional FFT layout.
    Independent real/imaginary canonical coordinates have unit mass:
    X_half=sqrt(w)*(Q_real+i Q_imag), Y_half=(P_real+i P_imag)/sqrt(w).
    """

    nonlocal_phonons = True

    def __init__(self, ph_freq, temperature, gmat, distribution="Boltzmann", use_gauge_phase=False):
        super().__init__(ph_freq, temperature, gmat, distribution)
        self.w_half, self.w = self.ph_freq.copy(), None
        self.use_gauge_phase = bool(use_gauge_phase)

    def initialize_position_and_momentum(self, nx, ny, ntraj, rng):
        self.nx, self.ny, self.ntraj = int(nx), int(ny), int(ntraj)
        if self.w_half.shape[1] != nx*ny//2:
            raise ValueError("nonlocal ph_freq requires exactly one entry per MP half-grid point")
        (self.w_full, self.q_half, self.q_partner, self.half_ij,
         self.partner_ij, self.qgrid_full) = get_mp_qmesh_info(nx, ny, self.w_half)
        self.rgrids = np.array([[x, y] for x in range(nx) for y in range(ny)])
        phase = np.einsum("ra,xya->rxy", self.rgrids, self.qgrid_full)
        self.expiqr = np.exp(2j*np.pi*phase)/np.sqrt(nx*ny)
        angles = np.random.default_rng(0).uniform(0, 2*np.pi, (self.nmodes, len(self.q_half)))
        self.gauge_phase_half = np.exp(1j*angles)
        hi, hj = self.half_ij.T
        pi, pj = self.partner_ij.T
        self.gauge_phase = np.zeros((self.nmodes, nx, ny), complex)
        self.gauge_phase[:, hi, hj] = self.gauge_phase_half
        self.gauge_phase[:, pi, pj] = self.gauge_phase_half.conj()
        shape = self.nmodes, ntraj, len(self.q_half)
        if self._supplied_samples is not None:
            if self._supplied_samples[0].shape != shape:
                raise ValueError(f"explicit nonlocal samples must have shape {shape}")
            self.q0_half, self.p0_half = (x.copy() for x in self._supplied_samples)
        else:
            qr, qi, pr, pi_values = (np.zeros(shape) for _ in range(4))
            for mode in range(self.nmodes):
                for iq, w in enumerate(self.w_half[mode]):
                    if self.distribution == "Boltzmann":
                        psigma, qsigma = np.sqrt(1/self.beta)/np.sqrt(2), np.sqrt(1/(self.beta*w*w))/np.sqrt(2)
                        pr[mode, :, iq] = rng.normal(0, psigma, ntraj)
                        pi_values[mode, :, iq] = rng.normal(0, psigma, ntraj)
                        qr[mode, :, iq] = rng.normal(0, qsigma, ntraj)
                        qi[mode, :, iq] = rng.normal(0, qsigma, ntraj)
                    else:
                        sigma = np.sqrt(1/np.tanh(self.beta*w/2))/np.sqrt(2)
                        qr[mode, :, iq] = rng.normal(0, sigma, ntraj)
                        qi[mode, :, iq] = rng.normal(0, sigma, ntraj)
                        pr[mode, :, iq] = rng.normal(0, sigma, ntraj)
                        pi_values[mode, :, iq] = rng.normal(0, sigma, ntraj)
            self.q0_half, self.p0_half = qr+1j*qi, pr+1j*pi_values
            if self.distribution == "Boltzmann":
                self.q0_half *= np.sqrt(2*self.w_half[:, None, :])
                self.p0_half *= np.sqrt(2/self.w_half[:, None, :])
        self.q0_full = self.reciprocal_half_to_full(self.q0_half)
        self.p0_full = self.reciprocal_half_to_full(self.p0_half)
        self.update_position(0.)

    def reciprocal_half_to_full(self, half):
        full = np.zeros((self.nmodes, self.ntraj, self.nx, self.ny), complex)
        hi, hj = self.half_ij.T
        pi, pj = self.partner_ij.T
        full[:, :, hi, hj], full[:, :, pi, pj] = half, half.conj()
        return full

    def position_reciprocal_to_real(self, reciprocal):
        gauge = self.gauge_phase if self.use_gauge_phase else np.ones_like(self.gauge_phase)
        field = np.einsum("utxy,rxy,uxy->utr", reciprocal, self.expiqr, gauge)
        if not np.allclose(field.imag, 0., atol=1e-10):
            raise ValueError("half-grid conjugation failed to produce a real oscillator field")
        return field.real

    def update_position(self, t):
        reciprocal = (self.q0_full*np.cos(self.w_full[:, None]*t)
                      + self.p0_full*np.sin(self.w_full[:, None]*t))
        self.qfield = self.position_reciprocal_to_real(reciprocal)

    def canonical_initial(self):
        if self.nmodes == 0:
            return np.zeros((self.ntraj, 1)), np.zeros((self.ntraj, 1))
        q = np.stack((self.q0_half.real, self.q0_half.imag), axis=-1)/np.sqrt(self.w_half[:, None, :, None])
        p = np.stack((self.p0_half.real, self.p0_half.imag), axis=-1)*np.sqrt(self.w_half[:, None, :, None])
        return (q.transpose(1, 0, 2, 3).reshape(self.ntraj, -1),
                p.transpose(1, 0, 2, 3).reshape(self.ntraj, -1))

    def native_map(self):
        hi, hj = self.half_ij.T
        return {"frequencies": self.w_half,
                "canonical_frequencies": np.repeat(self.w_half.reshape(-1), 2)
                if self.nmodes else np.zeros(1),
                "phase": self.expiqr[:, hi, hj],
                "gauge": self.gauge_phase_half if self.use_gauge_phase else np.ones_like(self.gauge_phase_half)}

    def initial_samples(self):
        return self.q0_half.copy(), self.p0_half.copy()


class QuantumPhononBath:
    """Local identical-site LF bath, explicitly separate from nuclear motion."""

    def __init__(self, ph_freq, gmat, temperature, band_narrow_only=False):
        self.w, coupling = np.asarray(ph_freq, float), np.asarray(gmat)
        if self.w.ndim != 1 or coupling.shape != self.w.shape or not np.isfinite(self.w).all():
            raise ValueError("quantum frequencies/couplings must be finite matching vectors")
        if np.any(self.w <= 0) or not np.isfinite(coupling).all():
            raise ValueError("quantum frequencies must be positive and couplings finite")
        self.couplings = coupling.copy()
        self.g_dimless = coupling/self.w
        self.beta = 1/_temperature(temperature)
        self.phi0 = np.sum(np.abs(self.g_dimless)**2/np.tanh(self.beta*self.w/2))
        self.phit = self.phi0
        self.polaron_prefactor = np.exp(-self.phi0)
        self.exponents = np.arange(-2, 3)
        # Historical inspection view; separate factors may overflow although
        # their combined native LF exponent is finite. Dynamics never multiplies
        # these separately by an underflowed static bath factor.
        with np.errstate(over="ignore", under="ignore"):
            self.sector_weights = np.exp(-self.exponents*self.phi0)
        self.band_narrow_only = bool(band_narrow_only)

    def update_phit(self, t):
        self.phit = np.sum(np.abs(self.g_dimless)**2 * (
            np.cos(self.w*t)/np.tanh(self.beta*self.w/2) - 1j*np.sin(self.w*t)))
        with np.errstate(over="ignore", under="ignore"):
            self.sector_weights = np.exp(-self.exponents*self.phit)


def build_phonon_baths(ph_freq, gmat, cpa_cutoff, temperature, distribution,
                      nonlocal_phonons=False, use_gauge_phase=False):
    """Preserve the established mean-frequency split and local LF extraction.

    Quantum couplings are taken from origin/origin orbital (0,0), exactly as
    in the original local identical-site approximation. This does not derive
    a nonlocal or site-dependent quantum-bath theory.
    """
    frequencies = np.asarray(ph_freq, float)
    if frequencies.ndim != 2 or not np.isfinite(cpa_cutoff):
        raise ValueError("ph_freq must be two-dimensional and cutoff finite")
    means = frequencies.mean(axis=1)
    classical = means <= cpa_cutoff
    blocks = {de: {dp: np.asarray(value)[..., classical].copy() for dp, value in nested.items()}
              for de, nested in gmat.items()}
    bath_type = ClassicalPhononNonlocal if nonlocal_phonons else ClassicPhononBath
    options = {"use_gauge_phase": use_gauge_phase} if nonlocal_phonons else {}
    bath = bath_type(frequencies[classical], temperature, blocks, distribution, **options)
    quantum = None
    if np.any(~classical):
        try:
            coupling = np.asarray(gmat[(0, 0)][(0, 0)])[0, 0, ~classical]
        except KeyError as exc:
            raise ValueError("local LF modes require origin-cell diagonal EPC data") from exc
        quantum = QuantumPhononBath(means[~classical], coupling, temperature)
    return bath, quantum
