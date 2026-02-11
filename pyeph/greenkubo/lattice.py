import numpy

class BravaisLattice2D:
    def __init__(self, nx, ny, ncenter, wcenter_pos, cell_vecs=None):
        self.nx = nx
        self.ny = ny
        self.ncenter = ncenter
        self.nsites  = ny * nx * ncenter
        self.ncells = nx * ny
        wcenter_pos = numpy.asarray(wcenter_pos, dtype=numpy.float64)
        wcenter_pos = wcenter_pos.reshape(ncenter, 2)
        self.wcenter_pos = wcenter_pos
        
        if cell_vecs is None:
            cell_vecs = numpy.array([[1.0, 0.0], [0.0, 1.0]])
        else:
            assert cell_vecs.shape == (2, 2)
        self.a1x, self.a1y = cell_vecs[0, 0], cell_vecs[0, 1]
        self.a2x, self.a2y = cell_vecs[1, 0], cell_vecs[1, 1]
        
        # cell index grid once
        rx = numpy.arange(nx, dtype=numpy.int64)
        ry = numpy.arange(ny, dtype=numpy.int64)
        self.rxs, self.rys = numpy.meshgrid(rx, ry, indexing='xy') # shape (ny, nx), (ny, nx)
        self.cell_idx = (self.rys * nx + self.rxs).ravel() # shape (ncells,)
        