from pyeph.greenkubo.lattice import BravaisLattice2D
from pyeph.greenkubo.hamiltonian import ElectronPhononHamiltonian
from pyeph.greenkubo.phonon import ClassicPhononBath, QuantumPhononBath
from pyeph.greenkubo.propagator import UnitaryPropagator
from pyeph.greenkubo.mpi_random import MPIRandomContext
from pyeph.utils.logger import logger
from pyeph.greenkubo.analysis import merge_outputs 
from pyeph.greenkubo.utils import get_map

import h5py
from pathlib import Path
import numpy
import time

class GreenKuboSimulation:
    def __init__(
        self,
        lattice: BravaisLattice2D,
        ham: ElectronPhononHamiltonian,
        classic_ph: ClassicPhononBath,
        quantum_ph: QuantumPhononBath,
        propagator: UnitaryPropagator,
        base_seed: int = 1120
    ):
        self.mpi_handler = MPIRandomContext(base_seed=base_seed)
        self.lattice = lattice
        self.ham = ham
        self.classic_ph = classic_ph
        self.quantum_ph = quantum_ph
        self.propagator = propagator
        self.build()
        
        self.is_1d_along_x = self.lattice.ny == 1
        self.is_1d_along_y = self.lattice.nx == 1
        
        self.current_x = []
        self.current_y = []
        self.current_times = []
        
        self.time_step = self.propagator.time_step
        self.total_time = self.propagator.total_time
        self.n_time_steps = len(self.propagator.time_range)
        #TODO: we can add checkpoint feature to restart calculation.

    def build(self):
        logger.info("Initializing phonon positions and momenta")
        self.classic_ph.initialize_position_and_momentum(
            self.lattice.nx, self.lattice.ny,
            self.propagator.ntraj, self.mpi_handler.rng
            )
        logger.info("Initializing ep-variation matrix")
        
        # build polaron transformation
        if self.quantum_ph is not None:
            self.propagator.build(self.ham, self.quantum_ph)
        self.polaron_prefactor = self.propagator.polaron_prefactor
        # self.ham.heps = self.ham.build_ep_variation_matrix(self.classic_ph.qfield, polaron_prefactor=self.polaron_prefactor)
        self.ham.heps = self.ham.build_ep_variation_matrix(self.classic_ph.qfield)
        # map = get_map(self.lattice.nx, self.lattice.ny)
        # numpy.save("he.npy", self.ham.heps[0].toarray()[:, map][map, :])
        logger.info("Initializing current operator")
        jx_0, jy_0 = self.ham.build_jx_jy(self.ham.heps)
        logger.info("Initializing density matrix")
        hep_polaron_transform = [hep * self.polaron_prefactor for hep in self.ham.heps]
        self.propagator.initialize_density_matrix(hep_polaron_transform, jx_0, jy_0)

    def run(self, dump_dir, dump_interval=None):
        """
        Evolve the system and collect current autocorrelation data.

        Parameters
        ----------
        dump_dir : str | Path | None
            When provided, gathered data on the root rank is flushed
            periodically to compressed ``.h5`` chunks inside this directory.
        dump_interval : int | None
            Number of simulation steps between dumps. If ``None``, data is
            only dumped once at the end of the run.
            Clear in-memory buffers of current data after each dump to keep the memory
            footprint small.

        """
        ntraj_per_rank = self.propagator.ntraj
        comm_size = self.mpi_handler.size
        job_start_time = time.time()
        chunk_start_time = time.time()
        self.initialize_dump(dump_dir, dump_interval)
        
        logger.info(
            f"Running Green-Kubo simulation with {ntraj_per_rank} trajectories/rank on {comm_size} ranks."
        )
        
        logger.info(f"Total steps: {self.n_time_steps}, each step per block: {self.dump_interval}")
        jx_t, jy_t = self.ham.build_jx_jy(self.ham.heps)
        ctx, cty = self.propagator.calculate_current(jx_t, jy_t)
        self.ctx_local = [ctx]
        self.cty_local = [cty]
        
        time_evolve = 0.0
        time_build_jx_jy = 0.0
        time_compute_current = 0.0
        
        for step_idx in range(1, self.n_time_steps):
            start_time = time.time()
            self.propagator.evolve(self.ham, self.classic_ph, self.quantum_ph)
            # map = get_map(self.lattice.nx, self.lattice.ny)
            # hep0 = self.ham.heps[0].toarray()[:, map][map, :]
            # numpy.save("he.npy", hep0)
            time_evolve += time.time() - start_time
            
            start_time = time.time()
            jx_t, jy_t = self.ham.build_jx_jy(self.ham.heps)
            time_build_jx_jy += time.time() - start_time
            
            start_time = time.time()
            ctx, cty = self.propagator.calculate_current(jx_t, jy_t)
            time_compute_current += time.time() - start_time
            self.ctx_local.append(ctx)
            self.cty_local.append(cty)
            
            if (step_idx+1) % self.dump_interval == 0:
                chunk_time = time.time() - chunk_start_time
                logger.info(
                    f"Step {step_idx}/{self.n_time_steps}, chunk {self._chunk_index} finished "
                    f"({self.dump_interval} steps/chunk, took {chunk_time:.2f}s)"
                )
                logger.info(
                    f"Timing breakdown per step: "
                    f"Evolve={time_evolve/self.dump_interval:.3f}s, "
                    f"Build J={time_build_jx_jy/self.dump_interval:.3f}s, "
                    f"Compute current={time_compute_current/self.dump_interval:.3f}s"
                )
                chunk_start_time = time.time()
                time_evolve = 0.0
                time_build_jx_jy = 0.0
                time_compute_current = 0.0
                self.dump(step_idx)
                self._chunk_index += 1
        
        if not self.dump_final:
            self.dump(step_idx)
        logger.info("Simulation on root rank finished, wait for data collection and cleanup")
        self.mpi_handler.barrier()
        if self.mpi_handler.rank == 0:
            axes = []
            if not self.is_1d_along_y:
                axes.append("x")
            if not self.is_1d_along_x:
                axes.append("y")
            merge_outputs(self.dump_dir, axes, comm_size, safe_mode=False)
        self.mpi_handler.barrier()
        self.dump_fname.unlink()
        logger.info(f"All jobs finished, time cost: {time.time() - job_start_time:.2f} seconds")
    
    def initialize_dump(self, dump_dir, dump_interval):
        self.dump_dir = Path(dump_dir)
        if self.mpi_handler.rank == 0:
            self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.mpi_handler.barrier()
        self.dump_fname = self.dump_dir / f"currents_{self.mpi_handler.rank}.h5"
        with h5py.File(self.dump_fname, "w") as fa:
            fa.attrs["total_time"] = self.total_time
            fa.attrs["time_step"] = self.time_step
            if not self.is_1d_along_y:
                fa.create_dataset("current_x", shape=(self.n_time_steps,), dtype=numpy.complex128)
            if not self.is_1d_along_x:
                fa.create_dataset("current_y", shape=(self.n_time_steps,), dtype=numpy.complex128)
            fa.attrs["current_step"] = 0

        self._chunk_index = 0
        self.dump_interval = dump_interval if dump_interval is not None else self.n_time_steps
        self.dump_final = False
        
    def dump(self, step_idx):
        with h5py.File(self.dump_fname, "a") as fa:
            if not self.is_1d_along_y:
                curx = numpy.asarray(self.ctx_local, dtype=numpy.complex128).mean(axis=1)
                s = int(fa.attrs["current_step"])
                e = s + len(curx)
                fa["current_x"][s:e] = curx
            if not self.is_1d_along_x:
                cury = numpy.asarray(self.cty_local, dtype=numpy.complex128).mean(axis=1)
                s = int(fa.attrs["current_step"])
                e = s + len(cury)
                fa["current_y"][s:e] = cury
            assert e == step_idx + 1
            fa.attrs["current_step"] = e
        
        self.ctx_local.clear()
        self.cty_local.clear()
        
        if step_idx == self.n_time_steps - 1:
            self.dump_final = True
