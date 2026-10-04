"""Historical module alias; both names use the indexed native EPC builder."""

from .hamiltonian import ElectronPhononHamiltonian, validate_displacement_data

__all__ = ["ElectronPhononHamiltonian", "validate_displacement_data"]
