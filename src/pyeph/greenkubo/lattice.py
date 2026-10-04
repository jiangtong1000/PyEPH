"""Host lattice topology with historical cell and site ordering.

The established PyEPH convention is site=(y*nx+x)*ncenter+orbital. Wannier
centres and row cell vectors are Cartesian in the same chosen length unit.
"""

import numpy as np

from pyeph.core.units import UnitSystem


class BravaisLattice2D:
    def __init__(self, nx, ny, ncenter, wcenter_pos, cell_vecs=None, *, unit_system=None):
        if any(not isinstance(n, (int, np.integer)) or n < 1 for n in (nx, ny, ncenter)):
            raise ValueError("lattice dimensions and ncenter must be positive integers")
        self.nx, self.ny, self.ncenter = int(nx), int(ny), int(ncenter)
        self.ncells, self.nsites = self.nx*self.ny, self.nx*self.ny*self.ncenter
        self.wcenter_pos = np.asarray(wcenter_pos, dtype=float).reshape(self.ncenter, 2).copy()
        self.cell_vecs = np.eye(2) if cell_vecs is None else np.asarray(cell_vecs, dtype=float).copy()
        if self.cell_vecs.shape != (2, 2) or not np.isfinite(self.cell_vecs).all():
            raise ValueError("cell_vecs must be a finite (2,2) row-vector matrix")
        if abs(np.linalg.det(self.cell_vecs)) < 1e-12 or not np.isfinite(self.wcenter_pos).all():
            raise ValueError("lattice vectors must be independent and centres finite")
        self.a1x, self.a1y = self.cell_vecs[0]
        self.a2x, self.a2y = self.cell_vecs[1]
        self.rxs, self.rys = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
        self.cell_idx = (self.rys*nx+self.rxs).ravel()
        self.unit_system = unit_system or UnitSystem()
        if not isinstance(self.unit_system, UnitSystem):
            raise TypeError("unit_system must be a UnitSystem")
        self.unit_scale_known = unit_system is not None
        for array in (self.wcenter_pos, self.cell_vecs, self.rxs, self.rys, self.cell_idx):
            array.flags.writeable = False

    def shifted_cells(self, dx, dy):
        return (((self.rys + dy) % self.ny)*self.nx + (self.rxs + dx) % self.nx).ravel()

    def minimum_displacement(self, row, column):
        """Antisymmetric minimum-image displacement for a fixed indexed edge.

        Tie-breaking retains the original no-image-first convention for the
        canonical row<column direction. The reverse is its exact negative.
        """
        if row == column:
            return np.zeros(2)
        if row > column:
            return -self.minimum_displacement(column, row)
        source_cell, target_cell = row // self.ncenter, column // self.ncenter
        delta = np.array([target_cell % self.nx-source_cell % self.nx,
                          target_cell // self.nx-source_cell // self.nx])
        centers = self.wcenter_pos[column % self.ncenter] - self.wcenter_pos[row % self.ncenter]
        candidates = np.array([
            (delta - [sx, sy]) @ self.cell_vecs + centers
            for sx in (0, -self.nx, self.nx) for sy in (0, -self.ny, self.ny)
        ])
        return candidates[np.argmin(np.sum(candidates**2, axis=1))]
