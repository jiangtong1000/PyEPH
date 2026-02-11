"""
We will minimize the free energy:
A=-\beta^{-1} \ln Trace e^{-\beta \tilde{H}_e}
A = -\beta^{-1} \ln \sum_k e^{-\beta \tilde{\epsilon}_k}
"""
import numpy
import scipy.optimize

def compute_f(w, g, H, beta, f0):
    """
    w: 1D array of float, phonon frequencies
    g: 1D array of float, electron-phonon coupling strengths
    H: 2D array of float, Hamiltonian
    beta: float, inverse temperature
    f0: 1D array, initial guess for the variational displacements
    
    return:
        1D array of float, optimized f values obtained by minimizing
        the free energy with a scipy.optimize-based search.
    """
    w = numpy.asarray(w, dtype=numpy.float64)
    g = numpy.asarray(g, dtype=numpy.float64)
    H = numpy.asarray(H, dtype=numpy.float64)

    if w.ndim != 1 or g.ndim != 1:
        raise ValueError("w and g must be 1D arrays.")
    if w.shape[0] != g.shape[0]:
        raise ValueError("w and g must have the same length.")
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be a square 2D array.")

    diag_idx = numpy.diag_indices(H.shape[0])

    def free_energy(f):
        f = numpy.asarray(f, dtype=numpy.float64)
        if f.shape != w.shape:
            raise ValueError("f must have the same length as w and g.")
        dw_factor = 1.0 / numpy.tanh(beta * w / 2.0)
        exp_gamma = numpy.exp(-numpy.sum(f**2 * dw_factor))
        tildeH = H * exp_gamma
        Ep = numpy.sum(w * (2.0 * g * f - f**2))
        tildeH = numpy.array(tildeH, copy=True)
        tildeH[diag_idx] = -Ep
        epsilon = numpy.linalg.eigh(tildeH)[0]
        return -1 / beta * numpy.log(numpy.sum(numpy.exp(-beta * (epsilon))))

    result = scipy.optimize.minimize(
        free_energy,
        numpy.asarray(f0, dtype=numpy.float64),
        method="BFGS",
        options={"gtol": 1e-6, "maxiter": 1000},
    )

    if not result.success:
        raise RuntimeError(f"Free-energy minimization failed: {result.message}")

    return numpy.asarray(result.x, dtype=numpy.float64)
