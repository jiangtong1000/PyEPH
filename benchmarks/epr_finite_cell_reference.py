"""Independent source-stencil assembly at arbitrary finite periodic cell meshes.

Stores raw versus projected matrix differences, native action/current/force
errors, and image-resolved current checks without constructing a dense EPC
Jacobian. No old PyEPH numerical output is an oracle.
"""

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from pyeph.adapters.epr import read_epr
from pyeph.core.contracts import ProbeContext, pure_state_weight


def direct_equations(data, mesh, q, c, charge):
    """Literal directed equations with independent host image indexing."""
    cells = np.array(list(np.ndindex(mesh)))
    nwan, nat = len(data.wannier_centers), len(data.masses)
    n = len(cells)*nwan
    raw = np.zeros((n, n), complex)
    current = np.zeros((3, n, n), complex)
    gradient, reference_gradient = np.zeros_like(q), np.zeros_like(q)
    displacement = (data.hopping_cells@data.cell
        +data.wannier_centers[data.hopping_orbitals[:, 1]]
        -data.wannier_centers[data.hopping_orbitals[:, 0]])
    for origin, cell in enumerate(cells):
        destination = np.ravel_multi_index(((cell+data.hopping_cells)%mesh).T, mesh)
        row = origin*nwan+data.hopping_orbitals[:, 0]
        column = destination*nwan+data.hopping_orbitals[:, 1]
        perturbed = np.ravel_multi_index(((cell+data.epc_cells)%mesh).T, mesh)
        nuclear = perturbed*nat+data.epc_atoms
        values = np.asarray(data.hopping_values, dtype=complex).copy()
        np.add.at(values, data.epc_channels, np.sum(data.epc_values*q[nuclear], axis=1))
        np.add.at(raw, (row, column), values)
        for axis in range(3):
            np.add.at(current[axis], (row, column), 1j*charge*displacement[:, axis]*values)
        weights = c[row].conj()*c[column]
        np.add.at(gradient, nuclear, np.real(weights[data.epc_channels, None]*data.epc_values))
        destination = np.ravel_multi_index(((cell+data.ifc_cells)%mesh).T, mesh)
        first = origin*nat+data.ifc_atoms[:, 0]
        second = destination*nat+data.ifc_atoms[:, 1]
        np.add.at(reference_gradient, first, np.einsum("tij,tj->ti", data.ifc_values, q[second]))
    return raw, .5*(current+current.conj().swapaxes(-1, -2)), gradient, reference_gradient


def run(path, mesh, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if not jax.config.x64_enabled:
        raise RuntimeError("this reference requires JAX_ENABLE_X64=1")
    source = read_epr(path, polar="short_range")
    compiled = source.compile_supercell(mesh, hermiticity="project")
    model, params = compiled.model, compiled.params
    rng = np.random.default_rng(914)
    q = .02*rng.normal(size=model.spec.system.q_shape)
    c = rng.normal(size=model.nstates)+1j*rng.normal(size=model.nstates)
    c /= np.linalg.norm(c)
    start = time.perf_counter()
    raw, current, gradient, reference = direct_equations(source.data, mesh, q, c, model.charge)
    direct_seconds = time.perf_counter()-start
    h = .5*(raw+raw.conj().T)
    actual = np.asarray(jax.jit(model.apply)(params, jnp.asarray(q), jnp.asarray(c)))
    actual_gradient = np.asarray(jax.jit(model.contract_gradient)(
        params, jnp.asarray(q), pure_state_weight(c)))
    actual_reference = np.asarray(jax.jit(model.reference_gradient)(params, jnp.asarray(q)))
    actual_current = np.array([model.probe_apply(params, ProbeContext(jnp.asarray(q)),
                                               f"current_{axis}", jnp.eye(model.nstates))
                               for axis in "xyz"])
    np.testing.assert_allclose(actual, h@c, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(actual_gradient, gradient, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(actual_reference, reference, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(actual_current, current, atol=2e-12, rtol=2e-12)
    report = dict(source=source.metadata, compilation=compiled.report,
        direct_assembly_seconds=direct_seconds,
        raw_matrix_max_antihermiticity=float(abs(raw-raw.conj().T).max()),
        matrix_projection_max_change=float(abs(raw-h).max()),
        native_action_max_error=float(abs(actual-h@c).max()),
        native_electronic_gradient_max_error=float(abs(actual_gradient-gradient).max()),
        native_reference_gradient_max_error=float(abs(actual_reference-reference).max()),
        native_peierls_current_max_error=float(abs(actual_current-current).max()),
        scope="finite-cell matrix/force/current equations; no trajectory or transport convergence")
    np.savez_compressed(output/"arrays.npz", q=q, c=c, raw=raw, projected=h, current=current,
                        gradient=gradient, reference_gradient=reference)
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epr", type=Path, required=True)
    parser.add_argument("--mesh", type=int, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.epr, tuple(args.mesh), args.output), indent=2))
