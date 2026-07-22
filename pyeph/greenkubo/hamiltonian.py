import numpy
import scipy.sparse
from pyeph.greenkubo.lattice import BravaisLattice2D
from pyeph.greenkubo.utils import write_hdf5_csr_matrix, write_hdf5_csr_matrix_list
from pyeph.utils.logger import logger

def validate_displacement_data(tmat, gmat):
    hop_keys = tuple(tmat.keys())
    ncenters = [numpy.size(tmat[key]) for key in hop_keys]
    if not all(ncenter == ncenters[0] for ncenter in ncenters):
        raise ValueError(
            "All hopping matrices must have the same number of centers"
        )
    ncenter = int(numpy.sqrt(ncenters[0]))
    assert ncenter**2 == ncenters[0]
    for key in hop_keys:
        tmat_key = numpy.array(tmat[key])
        tmat[key] = tmat_key.reshape(ncenter, ncenter)
    
    missing = [ikey for ikey in gmat.keys() if ikey not in hop_keys]
    if missing and numpy.unique(missing).size == 1 and missing[0] == (0, 0):
        # on-site is an exception
        missing = []
    if missing:
        raise ValueError(
            f"gmat contains displacements with no static hopping counterpart: {missing}"
        )
    if not all(((dx >= 0 and dy==0) or dy > 0) for dx, dy in hop_keys):
        raise ValueError(
            "tmat must only contain non-negative displacements; provide canonicalized data"
        )


class ElectronPhononHamiltonian:
    def __init__(self, tmat: dict, gmat: dict, lattice: BravaisLattice2D, debug=False):
        """
        tmat is a dictionary for different displacements D,
            each D contains a (ncenter, ncenter) matrix for hopping to the displaced cell
        """
        tmat = {
            key: array.real for key, array in tmat.items()
        }
        self.lattice = lattice
        validate_displacement_data(tmat, gmat)
        self.tmat = tmat
        self.gmat = gmat
        self._cell_shift_cache = {}
        self.build_static_hopping_matrix()
        self.debug = debug
    
    def _cell_index_with_shift(self, dx, dy):
        """
        Return flattened cell indices for lattice sites translated by (dx, dy),
        reusing cached values to avoid repeated mod/arange work.
        """
        key = (dx, dy)
        cache = self._cell_shift_cache
        if key not in cache:
            rxs = (self.lattice.rxs + dx) % self.lattice.nx
            rys = (self.lattice.rys + dy) % self.lattice.ny
            cache[key] = (rys * self.lattice.nx + rxs).ravel()
        return cache[key]
    
    def build_static_hopping_matrix(self, atol=1e-3):
        """
        Build the static electronic hopping matrix in 2D (nx by ny cells)
        h = sum_D sum_R sum_ij T(D)_{ij} a_{i,R}^\dagger a_{j,R+D}
        each cell contains ncenter wannier centers.
        tol: the threshold for pruning tiny contributions to the hopping matrix
        """
        logger.info("Building static hopping matrix")
        rows = []; cols = []; data = []
        hop_from_base = self.lattice.cell_idx * self.lattice.ncenter
        
        for (dx, dy), hopping_matrix in self.tmat.items():
            cell_idx_e = self._cell_index_with_shift(dx, dy)
            hop_to_base = cell_idx_e * self.lattice.ncenter
            # vectorized duplication over all center pairs
            for i in range(self.lattice.ncenter):
                hop_from = hop_from_base + i
                for j in range(self.lattice.ncenter):
                    v = hopping_matrix[i, j]
                    if numpy.abs(v) < atol: 
                        continue
                    hop_to = hop_to_base + j
                    rows.append(hop_from)
                    cols.append(hop_to)
                    data.append(numpy.full_like(hop_from, v, dtype=numpy.float64))
                    if not (dx == 0 and dy == 0):
                        rows.append(hop_to)
                        cols.append(hop_from)
                        data.append(numpy.full_like(hop_to, v, dtype=numpy.float64))

        rows = numpy.concatenate(rows)
        cols = numpy.concatenate(cols)
        data = numpy.concatenate(data)
        self.h_static = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(self.lattice.nsites, self.lattice.nsites)).tocsr()
        hstatic = self.h_static.toarray()
        assert numpy.allclose(hstatic, hstatic.T), "hstatic is not symmetric, something is wrong with the input tmat"
        row, col = self.h_static.nonzero()
        self.hopping_pairs = numpy.array(list(zip(row, col)))
        self.get_minimal_image_displacement()
    
    def get_minimal_image_displacement(self):
        site = numpy.arange(self.lattice.nsites, dtype=numpy.int64)
        cell_lin = site // self.lattice.ncenter # (0, ..., nx*ny-1)
        rx_site = (cell_lin % self.lattice.nx).astype(numpy.int64)
        ry_site = (cell_lin // self.lattice.nx).astype(numpy.int64)
        i_center = (site % self.lattice.ncenter).astype(numpy.int64)

        # intra-cell Wannier center coordinates (fractional)
        wx = self.lattice.wcenter_pos[:, 0]
        wy = self.lattice.wcenter_pos[:, 1]

        hep = self.h_static.tocoo(copy=False)
        r = hep.row
        c = hep.col
            
        def min_img(ri, ci):
            dist0 = numpy.inf
            drx0 = numpy.inf
            dry0 = numpy.inf
            for shift_x in [0, -self.lattice.nx, self.lattice.nx]:
                for shift_y in [0, -self.lattice.ny, self.lattice.ny]:
                    ri_x, ri_y = rx_site[ri] + shift_x, ry_site[ri] + shift_y
                    ci_x, ci_y = rx_site[ci], ry_site[ci]
                    dx_cell = ci_x - ri_x
                    dy_cell = ci_y - ri_y
                    drx = dx_cell * self.lattice.a1x + dy_cell * self.lattice.a2x + wx[i_center[ci]] - wx[i_center[ri]]
                    dry = dx_cell * self.lattice.a1y + dy_cell * self.lattice.a2y + wy[i_center[ci]] - wy[i_center[ri]]
                    dist = numpy.sqrt(drx**2 + dry**2)
                    if dist < dist0:
                        dist0 = dist
                        drx0 = drx
                        dry0 = dry
            return drx0, dry0
        
        # Handle when on-site energies are not zero        
        off_diag_mask = r != c
        r = r[off_diag_mask]
        c = c[off_diag_mask]
         
        drx = []
        dry = []
        for i in range(len(r)):
            ri = r[i]
            ci = c[i]
            drx0, dry0 = min_img(ri, ci)
            drx.append(drx0)
            dry.append(dry0)
        
        #TODO: such treatment only works for the case of zero onsite energies.
        self.drx = numpy.array(drx)
        self.dry = numpy.array(dry)
        
    def build_ep_variation_matrix(self, qfield, atol=1e-8):
        """
        build the ep-variation matrix hprime = h + hepc with same shape as from build_static_hopping_matrix,
        hepc = sum_De sum_Dp sum_R sum_ij g[De][Dp]_{ijv} Q_{R+Dp,v} a_{i,R}^\dagger a_{j,R+De}
        
        gmat is a nested dict, dict[(int,int) -> dict[(int,int) -> ndarray]]
        gmat[De][Dp] is (ncenter, ncenter, nmodes) with De=(dx_e,dy_e), Dp=(dx_p,dy_p).
        Contributes g_ijv(De,Dp) * Q_{R+Dp,v} to hopping (R,i)->(R+De,j).
        
        qfield : ndarray, shape (nmodes, ntraj, ny * nx)
            Classical phonon coordinates Q_{Rp,v} for each cell Rp=(x,y).
        
        atol: the threshold for pruning tiny contributions to the hopping matrix
        in unit of hopping
        
        return hep: a list of sparse matrices, each shape (nsites, nsites)
        """
        _, ntraj, _ = qfield.shape

        gq = {}
        # precompute g at q
        for (dx_e, dy_e), gmat_De in self.gmat.items():
            gq[(dx_e, dy_e)] = {}
            for (dx_p, dy_p), gmat_Dp in gmat_De.items():
                # (ncenter, ncenter, nmodes)(nmodes, ntraj, ncells) -> (ncenter, ncenter, ntraj, ncells)
                gq[(dx_e, dy_e)][(dx_p, dy_p)] = numpy.tensordot(gmat_Dp, qfield, axes=([-1], [0]))
                
        ncenter = self.lattice.ncenter
        ncells = self.lattice.ncells
        hop_from_base = self.lattice.cell_idx * ncenter
        rows = [[] for _ in range(ntraj)]
        cols = [[] for _ in range(ntraj)]
        data = [[] for _ in range(ntraj)]

        for (dx_e, dy_e), gmat_De in gq.items():
            hop_to_base = self._cell_index_with_shift(dx_e, dy_e) * ncenter
            for i in range(ncenter):
                hop_from = hop_from_base + i
                for j in range(ncenter):
                    hop_to = hop_to_base + j
                    accum = numpy.zeros((ntraj, ncells), dtype=numpy.float64)

                    for (dx_p, dy_p), gmat_Dp in gmat_De.items():
                        cell_idx_p = self._cell_index_with_shift(dx_p, dy_p)
                        local_accum = numpy.take(gmat_Dp[i, j], cell_idx_p, axis=-1)
                        accum += local_accum

                    nz_mask = numpy.abs(accum) > atol
                    if not numpy.any(nz_mask):
                        continue

                    for t in range(ntraj):
                        nz = nz_mask[t]
                        if not numpy.any(nz):
                            continue
                        rows[t].append(hop_from[nz])
                        cols[t].append(hop_to[nz])
                        data[t].append(accum[t, nz])
                        
                        if not (dx_e == 0 and dy_e == 0):
                            rows[t].append(hop_to[nz])
                            cols[t].append(hop_from[nz])
                            data[t].append(accum[t, nz])

        # one sparse matrix per trajectory
        hep_list = []
        for t in range(ntraj):
            if not rows[t]:
                hep_list.append(self.h_static.copy())
                continue
            r = numpy.concatenate(rows[t])
            c = numpy.concatenate(cols[t])
            d = numpy.concatenate(data[t])
            M = scipy.sparse.coo_matrix((d, (r, c)), shape=(self.lattice.nsites, self.lattice.nsites)).tocsr()
            # add the static hopping matrix
            hep = M + self.h_static
            hep_list.append(hep)
        
        # Debugging purpose
        # import h5py
        # from polar.greenkubo.utils import get_map
        # map = get_map(self.lattice.nx, self.lattice.ny)
        # hep = hep_list[0].toarray()
        # hep_reordered = hep[:, map][map, :]
        
        if self.debug:
            import h5py
            jx, jy = self.build_jx_jy(hep_list)

            with h5py.File("hep.h5", "w") as f:
                write_hdf5_csr_matrix(f, "hstatic", self.h_static)
                write_hdf5_csr_matrix_list(f, "heps", hep_list)
                write_hdf5_csr_matrix_list(f, "jx", jx)
                write_hdf5_csr_matrix_list(f, "jy", jy)
            exit()
        return hep_list

    def build_jx_jy(self, hep_list):
        """
        build the current operator in x and y directions
        Jx = i q / \hbar \sum_{m, n} (r_{m, x} - r_{n, x}) h_{m n} c_m^{\dagger} c_n
        Jy = i q / \hbar \sum_{m, n} (r_{m, y} - r_{n, y}) h_{m n} c_m^{\dagger} c_n
        I will omit i in this function, a minus sign will be needed in JJ autocorr.
        
        hep_list: list of csr matrices, shape (nsites, nsites)
        wcenter_pos: ndarray, shape (ncenter, 2)
            Position of wannier centers, in unit of cell vectors
        """
        Jx_list = []; Jy_list = []
        for hep in hep_list:
            hep = hep.tocoo(copy=False)

            # index arrays for nonzero entries in Hep
            r = hep.row
            c = hep.col
            data = hep.data
            
            # remove diagonal elements that has no contribution to the current
            off_diag_mask = r != c
            r = r[off_diag_mask]
            c = c[off_diag_mask]
            data = data[off_diag_mask]
            
            # build sparse matrices with same sparsity pattern as hep
            # This might be risky, but the probability is almost zero.
            Jx = scipy.sparse.coo_matrix((self.drx * data, (r, c)), shape=hep.shape, dtype=numpy.float64).tocsr()
            Jy = scipy.sparse.coo_matrix((self.dry * data, (r, c)), shape=hep.shape, dtype=numpy.float64).tocsr()

            for M in (Jx, Jy):
                M.sum_duplicates(); M.eliminate_zeros(); M.sort_indices()
            Jx_list.append(Jx)
            Jy_list.append(Jy)
        
        # Debugging purpose
        # import h5py
        # from polar.greenkubo.utils import get_map
        # map = get_map(self.lattice.nx, self.lattice.ny)
        # Jx = Jx_list[0].toarray()
        # Jy = Jy_list[0].toarray()
        # Jx_reordered = Jx[:, map][map, :]
        # Jy_reordered = Jy[:, map][map, :]
        # with h5py.File("J.h5", "w") as f:
        #     f.create_dataset("Jx", data=Jx_reordered)
        #     f.create_dataset("Jy", data=Jy_reordered)
        
        # remove polaron prefactor carried in hep
        # Jx_list = [Jx / polaron_prefactor for Jx in Jx_list]
        # Jy_list = [Jy / polaron_prefactor for Jy in Jy_list]
        return Jx_list, Jy_list
