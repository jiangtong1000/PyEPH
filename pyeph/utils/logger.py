"""Logger configuration for pyeph package"""

import logging
import sys
from typing import Optional

try:
    from mpi4py import MPI
    _HAS_MPI = True
except ImportError:
    _HAS_MPI = False


def get_mpi_rank() -> int:
    """Get MPI rank, returns 0 if MPI not available."""
    if _HAS_MPI:
        return MPI.COMM_WORLD.Get_rank()
    return 0


def get_mpi_size() -> int:
    """Get MPI size (number of processes), returns 1 if MPI not available."""
    if _HAS_MPI:
        return MPI.COMM_WORLD.Get_size()
    return 1


def get_mpi_comm():
    """Get MPI communicator, returns None if MPI not available."""
    if _HAS_MPI:
        return MPI.COMM_WORLD
    return None


def get_mpi_info() -> dict:
    """Get MPI information as a dictionary."""
    return {
        'has_mpi': _HAS_MPI,
        'rank': get_mpi_rank(),
        'size': get_mpi_size(),
        'comm': get_mpi_comm()
    }


def is_master_rank() -> bool:
    """Check if this is the master rank (rank 0)."""
    return get_mpi_rank() == 0


def setup_logger(
    name: str = "pyeph",
    level: str = "INFO",
    format_str: Optional[str] = None,
    stream: Optional[object] = None,
    master_only: bool = True
) -> logging.Logger:
    """
    Set up a logger for the pyeph package with MPI support.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_str: Custom format string
        stream: Output stream (default: sys.stdout)
        master_only: If True, only master rank (rank 0) will log
    
    Returns:
        Configured logger instance
    """
    if format_str is None:
        if _HAS_MPI:
            format_str = "[%(asctime)s] [rank %(rank)d] %(name)s.%(levelname)s: %(message)s"
        else:
            format_str = "[%(asctime)s] %(name)s.%(levelname)s: %(message)s"
    
    if stream is None:
        stream = sys.stdout
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Only add handler if master_only=False or if this is master rank
    if not master_only or is_master_rank():
        handler = logging.StreamHandler(stream)
        handler.setLevel(getattr(logging, level.upper()))
        
        # Create custom formatter that includes MPI rank
        if _HAS_MPI:
            class MPIFormatter(logging.Formatter):
                def format(self, record):
                    record.rank = get_mpi_rank()
                    return super().format(record)
            formatter = MPIFormatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")
        else:
            formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


# Default logger instance (master rank only)
default_logger = setup_logger()