from numpy.random import SeedSequence
import os
use_mpi = os.environ.get('USE_MPI', 'true').lower() == 'true'
if use_mpi:
    from mpi4py import MPI
    print(f'use_mpi = {use_mpi}')
else:
    from pyeph.utils.fake_mpi import MPI
    print('using fake mpi for debugging purposes')
    
import numpy

def spawn_rank_rng_sequence(base_seed, comm):
    rank = comm.Get_rank()
    if rank == 0:
        ss = SeedSequence(base_seed)
        stream = ss.spawn(comm.Get_size())
    else:
        stream = None
    stream = comm.bcast(stream, root=0)
    stream = stream[rank]
    return stream

class MPIRandomContext:
    def __init__(self, base_seed, comm=MPI.COMM_WORLD):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        seed_rank = spawn_rank_rng_sequence(base_seed, comm)
        self.rng = numpy.random.default_rng(seed_rank)

    def gather_data(self, data):
        """
        Gather arbitrary picklable data from all ranks to the root rank.

        Returns
        -------
        list | None
            On the root rank, returns a list ordered by rank id containing the
            gathered payloads. On non-root ranks the return value is None.
        """
        gathered = self.comm.gather(data, root=0)
        gathered = numpy.array(gathered) # (nrank, ...)
        if self.rank == 0:
            return gathered
        return None

    def barrier(self):
        """
        Synchronize all ranks participating in the communicator.
        """
        self.comm.Barrier()