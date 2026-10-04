"""Explicit host RNG compatibility; native states retain trajectory IDs.

Rank streams reproduce the original NumPy SeedSequence construction. They are
for historical fixtures, not a guarantee of repartition-independent samples.
"""

import numpy as np

from pyeph.utils.fake_mpi import MPI as SerialMPI
from pyeph.utils.logger import get_mpi_comm


def spawn_rank_rng_sequence(base_seed, comm):
    streams = np.random.SeedSequence(base_seed).spawn(comm.Get_size()) if comm.Get_rank() == 0 else None
    return comm.bcast(streams, root=0)[comm.Get_rank()]


class MPIRandomContext:
    def __init__(self, base_seed, comm=None):
        self.comm = comm if comm is not None else (get_mpi_comm() or SerialMPI.COMM_WORLD)
        self.rank, self.size = self.comm.Get_rank(), self.comm.Get_size()
        self.rng = np.random.default_rng(spawn_rank_rng_sequence(base_seed, self.comm))

    def gather_data(self, data):
        values = self.comm.gather(data, root=0)
        return np.asarray(values) if self.rank == 0 else None

    def barrier(self):
        self.comm.Barrier()
