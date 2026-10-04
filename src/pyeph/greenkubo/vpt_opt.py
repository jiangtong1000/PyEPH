"""Historical alias for the host variational-polaron optimizer."""

from .vpt import compute_f, _free_energy_fast_zero_diag, _validate_inputs

__all__ = ["compute_f", "_free_energy_fast_zero_diag", "_validate_inputs"]
