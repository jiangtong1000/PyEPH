"""Independent literal sum and cutoff regressions for 3D polar preprocessing."""

from types import SimpleNamespace

import numpy as np
import pytest

from pyeph.post_qe2pert.phonon_disp import PhononDispersion
from pyeph.post_qe2pert._polar import long_range_dynamical_matrix


def reciprocal_sum(phonons, qpoint, *, block_size=512):
    if not phonons.lpolar:
        return PhononDispersion.dyn_mat_longrange_3d(phonons, qpoint)
    params = phonons.polar_params
    return long_range_dynamical_matrix(
        qpoint, reciprocal_basis=phonons.bg, positions=phonons.tau,
        born_charges=params["zstar"], dielectric=params["epsil"],
        radius=params["nrx_ph"], alpha=params["polar_alpha"],
        cutoff=params["gmax"], volume=phonons.volume, block_size=block_size)


def fixture(complex_born=False):
    rng = np.random.default_rng(734)
    born = rng.normal(size=(4, 3, 3))
    if complex_born:
        born = born+1j*rng.normal(size=born.shape)
    return SimpleNamespace(lpolar=True, nat=4, volume=19.2,
        bg=np.array([[1.2, .1, .05], [.2, 1.3, -.2], [.0, .3, .9]]),
        tau=rng.normal(size=(4, 3)), polar_params=dict(nrx_ph=np.array([2, 3, 1]),
            polar_alpha=.7, gmax=3.4, epsil=np.array([[2., .1, 0.], [.1, 3., .2], [0., .2, 1.4]]),
            zstar=born))


def literal(phonons, q):
    p = phonons.polar_params
    result = np.zeros((3, 3, phonons.nat*(phonons.nat+1)//2), complex)
    radius = p["nrx_ph"]
    for cell in np.ndindex(tuple(2*radius+1)):
        g = phonons.bg.T @ (np.asarray(cell)-radius+q)
        squared = g @ p["epsil"] @ g
        if squared < 1e-14 or squared > p["gmax"]*4*p["polar_alpha"]:
            continue
        coefficient = np.exp(-squared/(4*p["polar_alpha"]))/squared
        pair = 0
        for b in range(phonons.nat):
            for a in range(b+1):
                phase = np.exp(2j*np.pi*g.dot(phonons.tau[a]-phonons.tau[b]))
                block = np.outer(g, g)*coefficient*phase
                result[:, :, pair] += p["zstar"][a] @ block @ p["zstar"][b].T
                pair += 1
    return result*8*np.pi/phonons.volume


@pytest.mark.parametrize("q", [np.zeros(3), np.array([.13, -.21, .37])])
@pytest.mark.parametrize("block_size", [1, 7, 512])
@pytest.mark.parametrize("complex_born", [False, True])
def test_bounded_contraction_matches_literal_sum(q, block_size, complex_born):
    phonons = fixture(complex_born)
    actual = reciprocal_sum(phonons, q, block_size=block_size)
    np.testing.assert_allclose(actual, literal(phonons, q), atol=2e-14, rtol=2e-14)


def test_translation_invariance_conjugate_q_and_disabled_or_empty_sum():
    phonons = fixture()
    q = np.array([.12, .23, -.16])
    reference = reciprocal_sum(phonons, q)
    np.testing.assert_allclose(reciprocal_sum(phonons, -q), reference.conj(), atol=2e-14)
    phonons.tau += [3., -.8, 1.2]
    np.testing.assert_allclose(reciprocal_sum(phonons, q), reference, atol=2e-14)
    phonons.polar_params["nrx_ph"] = np.zeros(3, dtype=int)
    np.testing.assert_array_equal(reciprocal_sum(phonons, np.zeros(3)), 0.)
    phonons.lpolar = False
    assert reciprocal_sum(phonons, q) is None


def test_large_exact_common_translation_preserves_pair_difference_phases():
    phonons = fixture()
    # Binary fractions remain exactly represented under this common shift.
    phonons.tau = np.array([[0., 0., 0.], [.5, -.25, .125],
                           [-.75, .125, .5], [.25, .375, -.625]])
    q = np.array([.12, .23, -.16])
    reference = literal(phonons, q)
    phonons.tau += [1e10, -2e10, 3e10]
    np.testing.assert_array_equal(literal(phonons, q), reference)
    np.testing.assert_allclose(reciprocal_sum(phonons, q), reference, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(PhononDispersion.dyn_mat_longrange_3d(phonons, q),
                               reference, atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize("block_size", [1, 7, 512])
@pytest.mark.parametrize("boundary", ["near_gamma", "upper"])
def test_scalar_cutoff_membership_is_preserved(boundary, block_size):
    """These SPD cases straddle a hard gate under the old einsum reduction."""
    phonons = SimpleNamespace(lpolar=True, nat=1, volume=3., bg=np.eye(3),
        tau=np.zeros((1, 3)), polar_params=dict(polar_alpha=1., gmax=14.,
            nrx_ph=np.array([2, 1, 2]), zstar=np.eye(3)[None]))
    if boundary == "near_gamma":
        q = np.array([3.5511997557010616e-8, -6.221648285765828e-8, -5.8950478654647375e-8])
        eps = np.array([[3.11082092017651, -.25969844681651993, .6319424489895286],
                        [-.25969844681651993, 2.5043904476974665, -1.6588552172303679],
                        [.6319424489895286, -1.6588552172303679, 2.8917701833069334]])
    else:
        # Fold q into the first reciprocal cell; G=(1,0,-2) lies at the gate.
        q = np.array([1.4448953656364032, .14292045958136101, -1.7506633029707839])
        q -= np.array([1., 0., -2.])
        eps = np.array([[11.276873970730438, 5.661708272064118, -1.5801506265504044],
                        [5.661708272064118, 14.804188143797477, .11932457348682193],
                        [-1.5801506265504044, .11932457348682193, 7.139705670871604]])
    phonons.polar_params["epsil"] = eps
    assert np.min(np.linalg.eigvalsh(eps)) > 0
    np.testing.assert_allclose(reciprocal_sum(phonons, q, block_size=block_size),
                               literal(phonons, q), rtol=2e-14, atol=2e-14)


def test_nan_dielectric_preserves_reference_invalid_result():
    phonons = fixture()
    phonons.polar_params["epsil"] = np.full((3, 3), np.nan)
    result = reciprocal_sum(phonons, np.array([.1, .2, .3]))
    reference = literal(phonons, np.array([.1, .2, .3]))
    assert np.isnan(reference).all()
    np.testing.assert_array_equal(np.isnan(result), np.isnan(reference))
