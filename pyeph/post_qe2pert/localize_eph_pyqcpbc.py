import jax
import jax.numpy as jnp
import numpy
import numpy as np
import h5py
from typing import Dict, Optional

from pyqcpbc.OPT import objective_function as qcpbc_objective
from pyqcpbc.OPT import minimizer as qcpbc_minimizer

from pyeph.post_qe2pert.wannier_phonon import assert_trs
from pyeph.utils.constants import ryd_to_mev
from pyeph.lib.setup_jax import configure_jax_backend
from pyeph.post_qe2pert.localize_eph import zero_out_negative_freqs, compute_density
from pyeph.utils.grid import get_shell_rph

configure_jax_backend()

def compute_eph_mat_real_space_pyqcpbc(
    eigvecs,
    freqs,
    masses,
    eph_raw,
    rph,
    re_ws_vectors,
    delta_r_vectors,
    q_hbz,
    q_minus,
    partner_hbz_for_minus,
    fname,
    rph_shell_radius=2,
    phfreq_cutoff=1.5 / ryd_to_mev,
    max_iter=200,
    tol=1e-9,
    algorithm="l_bfgs_ls"
):
    """
    Localize e-ph matrix elements by minimizing their spatial spread using pyqcpbc.
    Args:
        eigvecs: (nq, nmodes, nmodes)
        freqs: (nq, nmodes)
        masses: (natom,)
        eph_raw: (nband, nband, natom * 3, nre_ws, nq)
        rph: (nrph_ws, 3), the Wigner-Seitz lattice vectors
        delta_r_vectors: (n_delta_r, 3), the displacement vectors to minimize the objective
    """
    (nband, _, natom, _, nre_ws, nrph_ws) = eph_raw.shape
    assert_trs(eigvecs, q_hbz, q_minus, partner_hbz_for_minus)
    
    nq_hbz = q_hbz.shape[0]
    nq_minus = q_minus.shape[0]
    nq = nq_hbz + nq_minus
    nmodes = eigvecs.shape[1]
    assert eigvecs.shape == (nq, nmodes, nmodes)
    assert freqs.shape == (nq, nmodes)
    assert masses.shape == (natom,)
    assert rph.shape == (nrph_ws, 3)
    rph_shell = get_shell_rph(delta_r_vectors, rph_shell_radius)
    rph_shell = jnp.asarray(rph_shell, dtype=jnp.float64)
    re_ws_vectors = jnp.asarray(re_ws_vectors, dtype=jnp.float64)
    
    eigvecs, freqs = zero_out_negative_freqs(freqs, eigvecs, phfreq_cutoff)

    eigvecs = jnp.asarray(eigvecs, dtype=jnp.complex128)
    freqs = jnp.asarray(freqs, dtype=jnp.float64)
    masses = jnp.asarray(masses, dtype=jnp.float64)
    eph_raw = jnp.asarray(eph_raw, dtype=jnp.complex128)
    rph = jnp.asarray(rph, dtype=jnp.float64)
    delta_r_vectors = jnp.asarray(delta_r_vectors, dtype=jnp.float64)
    q_hbz = jnp.asarray(q_hbz, dtype=jnp.float64)
    q_minus = jnp.asarray(q_minus, dtype=jnp.float64)
    q_vectors = jnp.concatenate([q_hbz, q_minus], axis=0)
    partner_hbz_for_minus = jnp.asarray(partner_hbz_for_minus, dtype=jnp.int32)
    n_delta_r = delta_r_vectors.shape[0]
    n_rph_shell = rph_shell.shape[0]

    rng = numpy.random.default_rng()
    params_hbz = rng.standard_normal((nq_hbz, nmodes)) * 1e-3
    params_hbz = jnp.asarray(params_hbz, dtype=jnp.float64)

    mass_per_component = jnp.repeat(masses, 3)
    sqrt_mass_per_component = 1.0 / jnp.sqrt(mass_per_component)
    inv_sqrt_freqs = 1.0 / jnp.sqrt(2 * freqs)
    exp_iq_delta_r = jnp.exp(-1j * 2.0 * jnp.pi * delta_r_vectors @ q_vectors.T)
    exp_iq_r_shell = jnp.exp(-1j * 2.0 * jnp.pi * rph_shell @ q_vectors.T)
    exp_iqr = jnp.exp(1j * 2.0 * jnp.pi * rph @ q_vectors.T)
    eph_raw_rot = jnp.einsum("ijkaep, pq->ijkaeq", eph_raw, exp_iqr)
    eph_raw_rot = eph_raw_rot.reshape((nband, nband, natom * 3, nre_ws, nq))
    
    def build_phase_factors(param_array):
        phases_hbz = jnp.exp(1j * param_array)
        phases_minus = jnp.conj(phases_hbz[partner_hbz_for_minus])
        return jnp.concatenate([phases_hbz, phases_minus], axis=0)
    
    def compute_eph_real(param_array):
        phase_factors = build_phase_factors(param_array)
        rotated_eigvecs = jnp.einsum("qvu, qu, v, qu->qvu", eigvecs, phase_factors, sqrt_mass_per_component, inv_sqrt_freqs)
        eph_real = jnp.zeros((nband, nband, nre_ws, n_delta_r, nmodes), dtype=jnp.complex128)
        # only track the upper triangle since the e-ph matrix is symmetric
        for jw in range(nband):
            for iw in range(jw + 1):
                eph_mixed = jnp.einsum("veq, qvu->euq", eph_raw_rot[iw, jw], rotated_eigvecs)
                block = jnp.einsum("euq, rq->eru", eph_mixed, exp_iq_delta_r)
                eph_real = eph_real.at[iw, jw].set(block)
                eph_real = eph_real.at[jw, iw].set(block)
        return eph_real
    
    def compute_eph_real_shell(param_array):
        phase_factors = build_phase_factors(param_array)
        rotated_eigvecs = jnp.einsum("qvu, qu, v, qu->qvu", eigvecs, phase_factors, sqrt_mass_per_component, inv_sqrt_freqs)
        eph_real = jnp.zeros((nband, nband, nre_ws, n_rph_shell, nmodes), dtype=jnp.complex128)
        # only track the upper triangle since the e-ph matrix is symmetric
        for jw in range(nband):
            for iw in range(jw + 1):
                eph_mixed = jnp.einsum("veq, qvu->euq", eph_raw_rot[iw, jw], rotated_eigvecs)
                block = jnp.einsum("euq, rq->eru", eph_mixed, exp_iq_r_shell)
                eph_real = eph_real.at[iw, jw].set(block)
                eph_real = eph_real.at[jw, iw].set(block)
        return eph_real
    
    params_zeros = jnp.zeros_like(params_hbz)
    init_eph_real = compute_eph_real(params_zeros)
    assert jnp.allclose(init_eph_real.imag, 0.0)
    init_eph_real = jnp.abs(init_eph_real)
    total_g2 = jnp.sum(init_eph_real**2, axis=3)
    p_init, r_init = compute_density(init_eph_real, delta_r_vectors)
    with h5py.File(fname, "w") as f:
        f.create_dataset("init_eph_real", data=init_eph_real)
        f.create_dataset("init_p", data=p_init)
        f.create_dataset("init_r", data=r_init)
        f.create_dataset("rph_shell", data=rph_shell)
    
    def objective_variance(param_array):
        eph_real = compute_eph_real(param_array)
        p, r = compute_density(eph_real, delta_r_vectors)
        Er  = jnp.sum(p * r)
        Er2 = jnp.sum(p * r**2)
        return Er2 - Er * Er
    
    def objective(param_array):
        """
        the objective_variance minimizes the variance of the e-ph matrix elements
        which is translationally invariant
        we can assume the mean is zero so we are localizing to the origin directly
        """
        from pyeph.post_qe2pert.localize_eph import localize_reorganization_energy
        eph_real = compute_eph_real(param_array)
        loss = localize_reorganization_energy(eph_real, freqs, re_ws_vectors, delta_r_vectors)
        return loss
        # p, r = compute_density(eph_real, delta_r_vectors) # p: (nr, nmodes), r: (nr, )
        # second_moment = jnp.einsum('ru, r->u', p, r**2)
        # second_moment = jnp.sum(second_moment)
        # second_moment = jnp.sum(p * r**2)
        # second_moment = jnp.sum(p * r**2)
        # return second_moment
    
    @jax.jit
    def grad_function(param_array):
        grad = jax.grad(objective)(param_array)
        return grad
    
    def _pack_params(array):
        return np.asarray(array, dtype=np.float64).ravel()
    
    params_shape = params_hbz.shape
    def _unpack_params(vector):
        return jnp.asarray(vector.reshape(params_shape), dtype=jnp.float64)

    class PhononLocalizationObjective(qcpbc_objective):
        def __init__(self, initial_params_vec):
            super().__init__(initial_params_vec.size)
            self.params_vec = initial_params_vec.copy()
            self._origin_vec = self.params_vec.copy()
            self._current_value = None
            self._current_grad = None
            self._best_value = np.inf
            self._best_params_vec = self.params_vec.copy()

        def _evaluate(self):
            params = _unpack_params(self.params_vec)
            grad = grad_function(params)
            value = objective(params)
            grad_vec = np.asarray(grad).reshape(-1)
            self._current_value = value
            self._current_grad = grad_vec
            if value < self._best_value:
                self._best_value = value
                self._best_params_vec = self.params_vec.copy()
                with h5py.File(fname, "a") as f:
                    eph_real = compute_eph_real(params)
                    assert jnp.allclose(eph_real.imag, 0.0)
                    best_p, _ = compute_density(eph_real, delta_r_vectors)
                    eph_real_shell = compute_eph_real_shell(params)
                    assert jnp.allclose(eph_real_shell.imag, 0.0)
                    if "best_p" in f:
                        del f["best_p"]
                        del f['eph_real']
                        del f["eph_real_shell"]
                    f.create_dataset("best_p", data=best_p)
                    f.create_dataset("eph_real", data=eph_real)
                    f.create_dataset("eph_real_shell", data=eph_real_shell)
                    
        def get_value(self):
            self._evaluate()
            return self._current_value

        def grad(self):
            self._evaluate()
            return self._current_grad

        def precond(self, x, shift):
            return x

        def update_params(self, step):
            self.params_vec = self.params_vec + step.ravel()

        def save_new_origin(self):
            self._origin_vec = self.params_vec.copy()

        def back_to_origin(self):
            self.params_vec = self._origin_vec.copy()

        @property
        def best_params_vec(self):
            return self._best_params_vec

        @property
        def best_value(self):
            return self._best_value

    initial_vec = _pack_params(params_hbz)
    objective_wrapper = PhononLocalizationObjective(initial_vec)
    _ = objective_wrapper.get_value()

    solver = qcpbc_minimizer()
    solver.cfg("algorithm", algorithm)
    solver.cfg("maxiter", int(max_iter))
    solver.cfg("tol", float(tol))
    solver.run(objective_wrapper)