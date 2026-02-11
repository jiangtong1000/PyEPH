import numpy
import scipy.sparse.linalg
import scipy.linalg
from pyeph.greenkubo.estimator import (
    current_from_density_no_polaron,
    current_from_density_polaron,
    get_sectors_for_polaron_transform,
    _prepare_sector_arrays
)

class UnitaryPropagator:
    def __init__(self, nsites, ntraj_per_rank, time_step, total_time, temperature):
        self.nsites = nsites
        self.ntraj = ntraj_per_rank
        self.time_step = time_step
        self.total_time = total_time
        self.time_range = numpy.arange(0, total_time, time_step)
        self.beta = 1.0 / temperature
        self.time = 0.0

def _to_dense_list(mats, dtype=numpy.complex128):
    """
    Convert a list of sparse/array matrices to dense ndarrays.
    """
    dense = []
    for M in mats:
        if hasattr(M, "toarray"):
            dense.append(M.toarray().astype(dtype, copy=False))
        else:
            dense.append(numpy.asarray(M, dtype=dtype))
    return dense


def integrate_unitary_rk4(u_t0, h0, hmid, hfinal, dt, ntraj):
    for itraj in range(ntraj):
        ut_itraj = u_t0[itraj]
        k1 = -1.0j * h0[itraj] @ ut_itraj
        k2 = -1.0j * hmid[itraj] @ (ut_itraj + 0.5 * dt * k1)
        k3 = -1.0j * hmid[itraj] @ (ut_itraj + 0.5 * dt * k2)
        k4 = -1.0j * hfinal[itraj] @ (ut_itraj + dt * k3)
        u_t0[itraj] = ut_itraj + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return u_t0

def integrate_unitary_exact(u_t0, h0, dt, ntraj):
    for itraj in range(ntraj):
        u_t0[itraj] = scipy.sparse.linalg.expm_multiply(-1.0j * dt * h0[itraj], u_t0[itraj])
    return u_t0

class DensityMatrixUnitaryPropagator(UnitaryPropagator):
    def __init__(self, nsites, ntraj, time_step, total_time, temperature):
        super().__init__(nsites, ntraj, time_step, total_time, temperature)
        self.polaron_prefactor = 1.0
        self.polaron_transform = False
        self.calculate_current = self.calculate_current_no_polaron
        self.F0 = None
        self.sectors = None
        self.sec_weights = None
    
    def build(self, ham, quantum_ph):
        assert quantum_ph is not None
        self.polaron_transform = True
        if not quantum_ph.band_narrow_only:
            self.calculate_current = self.calculate_current_polaron
        
        F0 = {}
        for (i, j) in ham.hopping_pairs:
            for (k, l) in ham.hopping_pairs:
                key = (i, j, k, l)
                if key in F0:
                    continue
                coeff = (-2 + (i == j) + (k == l)) * quantum_ph.phi0
                F0[key] = numpy.exp(coeff)
        self.F0 = F0
        self.sectors = get_sectors_for_polaron_transform(ham.hopping_pairs)
        # Pre-flatten sectors/F0 for fast estimator calls.
        self.sector_quad_idx, self.sector_F0_vals, self.sector_offsets = _prepare_sector_arrays(self.sectors, self.F0)
        self.sec_weights = quantum_ph.sector_weights
        self.polaron_prefactor = quantum_ph.polaron_prefactor
        
    def initialize_density_matrix(self, heps, jx_0, jy_0):
        # Dense copies avoid repeated sparse->dense conversions later.
        self.jx_0 = _to_dense_list(jx_0)
        self.jy_0 = _to_dense_list(jy_0)
        self.j_rho0_x_T = [None] * self.ntraj
        self.j_rho0_y_T = [None] * self.ntraj
        self.rho0 = []
        for itraj in range(self.ntraj):
            hep_dense = heps[itraj].toarray()
            eigvals, eigvecs = scipy.linalg.eigh(hep_dense)
            rho = eigvecs @ numpy.diag(numpy.exp(-self.beta * eigvals)) @ eigvecs.T
            rho_0 = rho / rho.trace()
            self.rho0.append(rho_0)
            self.j_rho0_x_T[itraj] = (self.jx_0[itraj] @ rho_0).T
            self.j_rho0_y_T[itraj] = (self.jy_0[itraj] @ rho_0).T
        self.u_t = [numpy.eye(self.nsites, dtype=numpy.complex128) for _ in range(self.ntraj)]
        self.u_t = numpy.array(self.u_t)
        self.rho0 = numpy.array(self.rho0)
        
    def evolve(self, ham, classic_ph, quantum_ph):
        # rk4:
        classic_ph.update_position(self.time + 0.5 * self.time_step)
        hep_mid = ham.build_ep_variation_matrix(classic_ph.qfield)
        hep_mid = [hep * self.polaron_prefactor for hep in hep_mid]
        self.time = self.time + self.time_step
        classic_ph.update_position(self.time)
        if self.polaron_transform:
            quantum_ph.update_phit(self.time)
            self.sec_weights = quantum_ph.sector_weights
        ham.heps = ham.build_ep_variation_matrix(classic_ph.qfield)
        hep_final_polaron_transform = [hep * self.polaron_prefactor for hep in ham.heps]
        self.u_t = integrate_unitary_rk4(self.u_t, ham.heps, hep_mid, hep_final_polaron_transform, self.time_step, self.ntraj)
        '''
        # exact:
        classic_ph.update_position(self.time + self.time_step)
        hep_final = ham.build_ep_variation_matrix(classic_ph.qfield)
        self.u_t = integrate_unitary_exact(self.u_t, hep_final, self.time_step, self.ntraj)
        ham.heps = hep_final
        '''
        
    def calculate_current_no_polaron(self, jx_t, jy_t):
        ctx = numpy.empty(self.ntraj, dtype=numpy.complex128)
        cty = numpy.empty(self.ntraj, dtype=numpy.complex128)
        for itraj in range(self.ntraj):
            ctx[itraj] = current_from_density_no_polaron(
                self.j_rho0_x_T[itraj],
                self.u_t[itraj],
                jx_t[itraj]
            )
            cty[itraj] = current_from_density_no_polaron(
                self.j_rho0_y_T[itraj],
                self.u_t[itraj],
                jy_t[itraj]
            )
        return ctx, cty

    def calculate_current_polaron(self, jx_t, jy_t):
        # Debugging purpose
        # self.sec_weights = numpy.ones_like(self.sec_weights)
        # self.F0 = {key: 1.0 for key in self.F0.keys()}
        ctx = current_from_density_polaron(
            self.u_t,
            self.rho0,
            jx_t,
            self.jx_0,
            self.F0,
            self.sectors,
            self.sec_weights,
            quad_idx=self.sector_quad_idx,
            F0_vals=self.sector_F0_vals,
            sector_offsets=self.sector_offsets
        )
        cty = current_from_density_polaron(
            self.u_t,
            self.rho0,
            jy_t,
            self.jy_0,
            self.F0,
            self.sectors,
            self.sec_weights,
            quad_idx=self.sector_quad_idx,
            F0_vals=self.sector_F0_vals,
            sector_offsets=self.sector_offsets
        )
        
        '''
        # Debugging purpose
        from polar.greenkubo.utils import get_map
        import h5py
        map = get_map(4, 2)
        jx_t_dense = jx_t[0].toarray()
        jx_t_dense = jx_t_dense[:, map][map, :]
        jy_t_dense = jy_t[0].toarray()
        jy_t_dense = jy_t_dense[:, map][map, :]
        with h5py.File("newcode.h5", "a") as f:
            if "Jx" in f.keys():
                del f["Jx"]
                del f["Jy"]
                del f["U"]
            f.create_dataset("Jx", data=jx_t_dense)
            f.create_dataset("Jy", data=jy_t_dense)
            ut = self.u_t[0][:, map][map, :]
            f.create_dataset("U", data=ut)
        '''
        return ctx, cty