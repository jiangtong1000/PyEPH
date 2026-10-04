"""Historical logging/MPI names without import-time MPI initialization."""

import logging

from pyeph.post_qe2pert._support import (
    get_mpi_comm, get_mpi_info, get_mpi_rank, get_mpi_size, setup_logger,
)

logger = logging.getLogger("pyeph")


def is_master_rank():
    return get_mpi_rank() == 0


__all__ = ["logger", "setup_logger", "get_mpi_comm", "get_mpi_info",
           "get_mpi_rank", "get_mpi_size", "is_master_rank"]
