"""Explicit reduced-unit conversion at ingestion; kernels contain no unit objects."""

from dataclasses import dataclass

from scipy.constants import physical_constants

from pyeph.core._configuration import real_scalar

HARTREE_EV = physical_constants["Hartree energy in eV"][0]
BOHR_ANGSTROM = physical_constants["Bohr radius"][0] / 1e-10
ATOMIC_TIME_FS = physical_constants["atomic unit of time"][0] / 1e-15
KB_HARTREE_PER_K = physical_constants["Boltzmann constant in eV/K"][0] / HARTREE_EV


@dataclass(frozen=True)
class UnitSystem:
    """Energy/length scales relative to atomic units, with hbar=1 in kernels.

    Reduced time is atomic_time/energy_hartree and the corresponding mass unit
    is electron_mass/(energy_hartree*length_bohr**2). Temperatures in kernels are
    kB*T divided by the energy scale, rather than temperatures in Kelvin.
    """

    energy_hartree: float = 1.0
    length_bohr: float = 1.0

    def __post_init__(self):
        for name in ("energy_hartree", "length_bohr"):
            object.__setattr__(self, name, real_scalar(getattr(self, name), name))
        if any(x <= 0 for x in (self.energy_hartree, self.length_bohr)):
            raise ValueError("unit scales must be finite and positive")

    @classmethod
    def from_ev_angstrom(cls, energy_ev=1.0, length_angstrom=1.0):
        return cls(energy_ev / HARTREE_EV, length_angstrom / BOHR_ANGSTROM)

    @property
    def time_fs(self):
        return ATOMIC_TIME_FS / self.energy_hartree

    def temperature_from_kelvin(self, kelvin):
        return kelvin * KB_HARTREE_PER_K / self.energy_hartree

    def mass_from_electron_masses(self, mass):
        return mass * self.energy_hartree * self.length_bohr**2

    def time_from_fs(self, time):
        return time / self.time_fs

    def gradient_from_atomic(self, gradient):
        return gradient * self.length_bohr / self.energy_hartree
