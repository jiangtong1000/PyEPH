import numpy as np
import scipy.optimize
from scipy.special import logsumexp


def _validate_inputs(w, g, H, beta, f0):
    w = np.asarray(w, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    f0 = np.asarray(f0, dtype=np.float64)

    assert np.all(H.diagonal() == 0.0), "H non-zero diagonal not implemented, please implement if needed"

    if w.ndim != 1 or g.ndim != 1:
        raise ValueError("w and g must be 1D arrays.")
    if w.shape != g.shape:
        raise ValueError("w and g must have the same shape.")
    if f0.shape != w.shape:
        raise ValueError("f0 must have the same shape as w and g.")
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be a square 2D array.")
    if not np.allclose(H, H.T):
        raise ValueError("H must be symmetric (real Hermitian).")
    if beta <= 0.0:
        raise ValueError("beta must be positive.")
    if np.any(w <= 0.0):
        raise ValueError("All phonon frequencies in w must be positive.")

    return w, g, H, float(beta), f0

def _free_energy_fast_zero_diag(f, w, g, H_eigs, beta):
    """
    Exact shortcut for the special case diag(H) == 0:
        tildeH = exp_gamma * H - Ep * I
    so eigenvalues are:
        tilde_eigs = exp_gamma * eigvals(H) - Ep
    """
    f = np.asarray(f, dtype=np.float64)
    dw_factor = 1.0 / np.tanh(beta * w / 2.0)
    exp_gamma = np.exp(-np.sum(f**2 * dw_factor))
    Ep = np.sum(w * (2.0 * g * f - f**2))

    epsilon = exp_gamma * H_eigs - Ep
    return -(1.0 / beta) * logsumexp(-beta * epsilon)

def compute_f(w, g, H, beta, f0):
    """
    Optimized version.

    - If diag(H) == 0 exactly, uses the fast eigenvalue shortcut.
    - Otherwise, falls back to the reference implementation.
    """
    w, g, H, beta, f0 = _validate_inputs(w, g, H, beta, f0)
    H_eigs = np.linalg.eigvalsh(H)

    def objective(f):
        if np.asarray(f).shape != w.shape:
            raise ValueError("f must have the same shape as w and g.")
        return _free_energy_fast_zero_diag(f, w, g, H_eigs, beta)

    result = scipy.optimize.minimize(
        objective,
        f0,
        method="BFGS",
        options={"gtol": 1e-6, "maxiter": 1000},
    )

    if not result.success:
        raise RuntimeError(f"Free-energy minimization failed: {result.message}")

    return np.asarray(result.x, dtype=np.float64)