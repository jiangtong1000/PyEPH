"""Historical lattice Hamiltonian preparation, backed by native sparse terms.

Dictionary and site conventions follow PyEPH revision 6c4693ac (BSD-3-Clause).
This implementation fixes the old implicit COO-data/current-displacement match:
every current displacement belongs to a fixed (row,column) edge, including
edges contributed only by EPC and coordinates where a hopping becomes zero.
"""

from collections import defaultdict

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from pyeph.models.lattice_epc import LatticeEPCModel
from ._precision import require_legacy_precision


def _displacement(key):
    if len(key) != 2 or any(x != int(x) for x in key):
        raise ValueError("cell displacements must be integer pairs")
    return tuple(int(x) for x in key)


def validate_displacement_data(tmat, gmat):
    """Validate canonical electronic displacements; permit EPC-only edges."""
    for key in set(tmat) | set(gmat):
        x, y = _displacement(key)
        if not ((x >= 0 and y == 0) or y > 0):
            raise ValueError("electronic displacements must use the canonical half plane")
    for nested in gmat.values():
        for key in nested:
            _displacement(key)


class ElectronPhononHamiltonian:
    def __init__(self, tmat, gmat, lattice, debug=False):
        self.lattice, self.debug = lattice, bool(debug)
        validate_displacement_data(tmat, gmat)
        n = lattice.ncenter
        self.tmat = {tuple(k): np.asarray(v).reshape(n, n).copy() for k, v in tmat.items()}
        self.gmat = {tuple(k): {tuple(p): np.asarray(v).copy() for p, v in d.items()}
                     for k, d in gmat.items()}
        mode_counts = {v.shape[-1] for nested in self.gmat.values() for v in nested.values()}
        if len(mode_counts) > 1:
            raise ValueError("all EPC blocks must contain the same number of modes")
        self.nmodes = next(iter(mode_counts), 0)
        for key, value in self.tmat.items():
            self._validate_block(value, key, (n, n))
        for key, nested in self.gmat.items():
            for value in nested.values():
                if value.shape != (n, n, self.nmodes) or not np.isfinite(value).all():
                    raise ValueError("EPC blocks must be finite with shape (ncenter,ncenter,nmodes)")
        self.build_static_hopping_matrix()

    @staticmethod
    def _validate_block(value, key, shape):
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"hopping/EPC block must be finite with shape {shape}")
        if key == (0, 0) and not np.allclose(value, value.swapaxes(0, 1).conj(), atol=1e-12):
            raise ValueError("within-cell onsite/hopping blocks must be Hermitian")

    def _cell_index_with_shift(self, dx, dy):
        return self.lattice.shifted_cells(dx, dy)

    def build_static_hopping_matrix(self, atol=1e-3):
        """Compile once; only this host static-hopping cutoff is retained.

        EPC coefficients retain every nonzero term. No coordinate-dependent
        pruning occurs inside the differentiable native action.
        """
        lattice = self.lattice
        static = defaultdict(complex)
        terms = defaultdict(complex)
        n = lattice.ncenter
        for displacement, block in self.tmat.items():
            target = lattice.shifted_cells(*displacement)
            for i, j in np.ndindex(n, n):
                if abs(block[i, j]) < atol:
                    continue
                for cell in range(lattice.ncells):
                    r, c = cell*n+i, int(target[cell])*n+j
                    static[r, c] += block[i, j]
                    if displacement != (0, 0):
                        static[c, r] += block[i, j].conjugate()
        for displacement, nested in self.gmat.items():
            target = lattice.shifted_cells(*displacement)
            for phonon_displacement, block in nested.items():
                phonon_cells = lattice.shifted_cells(*phonon_displacement)
                for i, j, mode in zip(*np.nonzero(block)):
                    for cell in range(lattice.ncells):
                        r, c = cell*n+int(i), int(target[cell])*n+int(j)
                        f = int(mode)*lattice.ncells+int(phonon_cells[cell])
                        terms[r, c, f] += block[i, j, mode]
                        if displacement != (0, 0):
                            terms[c, r, f] += block[i, j, mode].conjugate()
        terms = {key: value for key, value in terms.items() if value != 0}
        edges = sorted(set(static) | {key[:2] for key in terms})
        edge_lookup = {key: i for i, key in enumerate(edges)}
        self.rows = np.array([e[0] for e in edges], dtype=np.int32)
        self.columns = np.array([e[1] for e in edges], dtype=np.int32)
        self.static_values = np.array([static[e] for e in edges], dtype=complex)
        ordered_terms = sorted(terms)
        self.term_edges = np.array([edge_lookup[k[:2]] for k in ordered_terms], dtype=np.int32)
        self.field_indices = np.array([k[2] for k in ordered_terms], dtype=np.int32)
        self.coefficients = np.array([terms[k] for k in ordered_terms], dtype=complex)
        self.displacements = np.array([lattice.minimum_displacement(*e) for e in edges]).reshape(-1, 2)
        self._edge_lookup = edge_lookup
        self.h_static = sp.csr_matrix((self.static_values, (self.rows, self.columns)),
                                      shape=(lattice.nsites, lattice.nsites))
        self.h_static.eliminate_zeros()
        # Union, not the sparsity pattern of one sample or only static hoppings.
        self.hopping_pairs = np.array(edges, dtype=np.int32).reshape(-1, 2)
        mask = self.rows != self.columns
        self.drx, self.dry = self.displacements[mask].T
        for array in (self.rows, self.columns, self.static_values, self.term_edges,
                      self.field_indices, self.coefficients, self.displacements, self.hopping_pairs):
            array.flags.writeable = False
        return self.h_static

    def get_minimal_image_displacement(self):
        return self.displacements

    def build_ep_variation_matrix(self, qfield, atol=1e-8):
        """Return SciPy CSR views for old callers; native evolution bypasses this.

        The historical ``atol`` argument is accepted but does not prune an
        evolving graph. This removes the old topology-dependent current bug.
        """
        fields = np.asarray(qfield)
        if fields.ndim != 3 or fields.shape[0] != self.nmodes or fields.shape[2] != self.lattice.ncells:
            raise ValueError("qfield must have shape (nmodes,ntraj,ncells)")
        matrices = []
        for field in fields.transpose(1, 0, 2).reshape(fields.shape[1], -1):
            values = self.static_values.copy()
            np.add.at(values, self.term_edges, self.coefficients*field[self.field_indices])
            matrix = sp.csr_matrix((values, (self.rows, self.columns)), shape=self.h_static.shape)
            matrix.eliminate_zeros()
            matrices.append(matrix)
        return matrices

    def build_jx_jy(self, hep_list):
        """Return historical currents with the factor i omitted, indexed safely."""
        currents = ([], [])
        for matrix in hep_list:
            coo = sp.coo_matrix(matrix)
            displacement = np.array([self.lattice.minimum_displacement(int(r), int(c))
                                     for r, c in zip(coo.row, coo.col)]).reshape(-1, 2)
            for axis in (0, 1):
                current = sp.csr_matrix((coo.data*displacement[:, axis], (coo.row, coo.col)), shape=coo.shape)
                current.eliminate_zeros()
                currents[axis].append(current)
        return currents

    def native(self, bath):
        """Compile an independent bath's canonical map into the shared model API."""
        require_legacy_precision()
        if bath.nmodes != self.nmodes:
            raise ValueError("classical bath and EPC mode counts differ")
        # Legacy host constructors sometimes carry unused trial EPC blocks.
        # Enforce the native Hamiltonian contract before any dynamics starts.
        for displacement, nested in self.gmat.items():
            for value in nested.values():
                self._validate_block(value, displacement,
                                     (self.lattice.ncenter, self.lattice.ncenter, self.nmodes))
        nonlocal_modes = getattr(bath, "nonlocal_phonons", False)
        model = LatticeEPCModel(self.lattice.nsites, self.nmodes, self.lattice.ncells,
                                self.lattice.ncells//2 if nonlocal_modes else 0,
                                nonlocal_modes, self.lattice.unit_system)
        params = {name: jnp.asarray(getattr(self, attr)) for name, attr in {
            "rows": "rows", "columns": "columns", "static": "static_values",
            "term_edges": "term_edges", "field_indices": "field_indices",
            "coefficients": "coefficients", "displacements": "displacements",
        }.items()}
        params.update({key: jnp.asarray(value) for key, value in bath.native_map().items()})
        model.validate_params(params)
        return model, params
