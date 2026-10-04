"""Independent checks of the FFT reference, not comparisons to its own steps."""

import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm

from benchmarks.mash2_wavepacket import ModifiedTully, lower_vectors, quantum_run, split_operator


def test_fft_reference_exact_plane_wave_and_constant_electronic_rotation():
    q = np.linspace(-np.pi, np.pi, 128, endpoint=False)
    momentum, mass, duration = 3., 2.3, 7.9
    spin = np.array([np.sqrt(.3), 1j*np.sqrt(.7)])
    psi = np.exp(1j*momentum*q)[:, None]*spin/np.sqrt(len(q))
    matrix = np.array([[.4, .17], [.17, -.4]])
    result, diagnostics = split_operator(psi, q, mass=mass, duration=duration, dt=.19,
                                        potential=lambda x: (.4, .17))
    exact_spin = expm(-1j*duration*matrix)@spin
    exact = (np.exp(1j*momentum*q-1j*duration*momentum**2/(2*mass))[:, None]
             * exact_spin/np.sqrt(len(q)))
    np.testing.assert_allclose(result, exact, atol=1e-14, rtol=1e-13)
    assert abs(np.sum(np.abs(result)**2)-1) < 2e-14
    assert diagnostics["steps"] == 42


def test_fft_reference_free_gaussian_position_and_momentum_moments():
    q = np.linspace(-50., 50., 2048, endpoint=False)
    dx = q[1]-q[0]
    gamma, p0, q0, mass, duration = .7, 1.3, -3., 2., 4.
    scalar = ((gamma/np.pi)**.25*np.sqrt(dx)
              * np.exp(-.5*gamma*(q-q0)**2+1j*p0*(q-q0)))
    psi = np.column_stack((scalar, np.zeros_like(scalar)))
    result, diagnostics = split_operator(psi, q, mass=mass, duration=duration, dt=.3,
                                        potential=lambda x: (0., 0.))
    probability = np.sum(np.abs(result)**2, axis=1)
    mean = q0+p0*duration/mass
    variance = (1+(gamma*duration/mass)**2)/(2*gamma)
    np.testing.assert_allclose([q@probability, ((q-mean)**2)@probability],
                               [mean, variance], atol=3e-13, rtol=1e-13)
    momentum = 2*np.pi*np.fft.fftfreq(len(q), dx)
    probability_p = np.sum(np.abs(np.fft.fft(result, axis=0, norm="ortho"))**2, axis=1)
    np.testing.assert_allclose([momentum@probability_p, ((momentum-p0)**2)@probability_p],
                               [p0, gamma/2], atol=1e-13, rtol=1e-13)
    assert diagnostics["sampled_max_outer_five_percent_probability"] < 1e-25


def test_modified_tully_model_uses_paper_tanh_and_continuous_lower_state():
    model = ModifiedTully()
    for q in (-15., -2., -.1, 0., .1, 2., 15.):
        z, coupling = .01*np.tanh(1.6*q), .005*np.exp(-q*q)
        matrix = np.array([[z, coupling], [coupling, -z]])
        np.testing.assert_allclose(model.dense(None, jnp.array([q])), matrix, atol=1e-15)
        lower = lower_vectors(q)
        np.testing.assert_allclose(matrix@lower, -np.hypot(z, coupling)*lower, atol=1e-17)
        np.testing.assert_allclose(lower@lower, 1., atol=2e-16)


def test_fft_reference_noncommuting_fourier_dvr_exponential_has_second_order_error():
    # Construct a small periodic Fourier-DVR Hamiltonian independently of FFT
    # stepping. The position-dependent potential does not commute with kinetic
    # energy, so this detects split order and missing/duplicated half steps.
    n, mass, duration = 40, 3., .6
    q = np.linspace(-np.pi, np.pi, n, endpoint=False)
    momentum = np.concatenate((np.arange(n//2), np.arange(-n//2, 0)))
    indices = np.arange(n)
    fourier = np.exp(-2j*np.pi*np.outer(indices, indices)/n)/np.sqrt(n)
    kinetic = fourier.conj().T @ ((momentum**2/(2*mass))[:, None]*fourier)
    hamiltonian = np.kron(kinetic, np.eye(2))

    def potential(x):
        return .3*np.cos(x)+.1*np.cos(2*x), .2*np.sin(x)+.15

    z, coupling = potential(q)
    for index in range(n):
        block = slice(2*index, 2*index+2)
        hamiltonian[block, block] += [[z[index], coupling[index]], [coupling[index], -z[index]]]
    envelope = np.exp(-2*np.sin((q-.3)/2)**2+1j*q)
    psi = envelope[:, None]*np.array([.6, .8j])
    psi /= np.linalg.norm(psi)
    exact = (expm(-1j*duration*hamiltonian) @ psi.ravel()).reshape(n, 2)
    errors = []
    for dt in (.06, .03, .015):
        result, _ = split_operator(psi, q, mass=mass, duration=duration, dt=dt,
                                   potential=potential)
        errors.append(np.linalg.norm(result-exact))
    orders = np.log2(np.asarray(errors[:-1])/errors[1:])
    assert np.all((orders > 1.98) & (orders < 2.02))
    assert errors[-1] < 1e-5


def test_quantum_probability_masses_and_densities_share_normalization():
    statistics, arrays = quantum_run("high", points=4096, duration=1.)
    for coordinate in ("q", "p"):
        spacing = statistics["dx" if coordinate == "q" else "dp"]
        mass, density = arrays["probability_"+coordinate], arrays["density_"+coordinate]
        np.testing.assert_allclose(density*spacing, mass, atol=1e-16)
        np.testing.assert_allclose(np.sum(density)*spacing, 1., atol=1e-13)
