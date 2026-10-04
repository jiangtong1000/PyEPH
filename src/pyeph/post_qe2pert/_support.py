"""Small, dependency-local helpers for the migrated QE2PERT preprocessing.

Conventions originate in jiangtong1000/PyEPH (BSD-3-Clause, see LICENSE).
MPI is optional and imported only when a preprocessing operation requests it.
Importing this module does not configure a JAX backend or initialize logging.
"""

import logging
import os
import sys

import numpy as np

# Preserve the original preprocessing conversion constants for numerical parity.
ryd_to_ev = 13.605698066
ryd_to_mev = ryd_to_ev * 1000
bohr_to_ang = 0.52917721092


def get_mpi_comm():
    """Return COMM_WORLD when enabled and available, otherwise run serially."""
    if os.environ.get("USE_MPI", "true").lower() != "true":
        return None
    try:
        from mpi4py import MPI
    except (ImportError, RuntimeError):
        return None
    return MPI.COMM_WORLD


def get_mpi_rank():
    comm = get_mpi_comm()
    return 0 if comm is None else comm.Get_rank()


def get_mpi_size():
    comm = get_mpi_comm()
    return 1 if comm is None else comm.Get_size()


def get_mpi_info():
    comm = get_mpi_comm()
    return {
        "has_mpi": comm is not None,
        "rank": 0 if comm is None else comm.Get_rank(),
        "size": 1 if comm is None else comm.Get_size(),
        "comm": comm,
    }


def setup_logger(name="pyeph", level="INFO", format_str=None, stream=None, master_only=True):
    """Create one local rank-aware handler without removing caller handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    # Remove only handlers created here when refreshing a preprocessing logger.
    for handler in list(logger.handlers):
        if getattr(handler, "_pyeph_preprocessing", False):
            logger.removeHandler(handler)
    rank = get_mpi_rank()
    if not master_only or rank == 0:
        handler = logging.StreamHandler(sys.stdout if stream is None else stream)
        handler._pyeph_preprocessing = True

        class RankFormatter(logging.Formatter):
            def format(self, record):
                record.rank = rank
                return super().format(record)

        handler.setFormatter(RankFormatter(
            format_str or "[%(asctime)s] %(name)s.%(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
    return logger


def get_shell_rph(rph, rph_shell):
    """Select displacement vectors within a Euclidean-radius shell."""
    rph = np.asarray(rph)
    return rph[np.linalg.norm(rph, axis=1) <= rph_shell]


def assert_trs(modes_all, q_hbz, q_minus, partner_hbz_for_minus):
    """Validate the original half-zone/conjugate-partner eigenvector contract."""
    modes_all = np.asarray(modes_all)
    identity = np.eye(modes_all.shape[-1])
    for iq in range(len(q_hbz)):
        assert np.allclose(modes_all[iq].conj().T @ modes_all[iq], identity)
        assert np.allclose(modes_all[iq] @ modes_all[iq].conj().T, identity)
    assert len(q_minus) == len(partner_hbz_for_minus)
    assert len(q_hbz) + len(q_minus) == len(modes_all)
    for iqm, partner in enumerate(partner_hbz_for_minus):
        assert 0 <= partner < len(q_hbz)
        assert np.allclose(modes_all[iqm + len(q_hbz)], modes_all[partner].conj())
