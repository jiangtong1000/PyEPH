"""Small immutable declarations; arrays and numerical parameters live elsewhere."""

from dataclasses import dataclass
from math import prod

from pyeph.core._configuration import integer_scalar


@dataclass(frozen=True)
class SystemSpec:
    """Fixed electronic labels and the shape of one trajectory's coordinates.

    All native kernels use hbar=1 in one consistent unit system. Cartesian and
    linearly transformed canonical coordinates are supported. The kinetic mass
    is supplied by the nuclear treatment, rather than inferred from array names.
    """

    nstates: int
    q_shape: tuple[int, ...]
    basis_id: str = "fixed"
    coordinate_kind: str = "cartesian"

    def __post_init__(self):
        object.__setattr__(self, "nstates", integer_scalar(self.nstates, "nstates"))
        try:
            shape = tuple(integer_scalar(n, "q_shape dimension") for n in self.q_shape)
        except TypeError as exc:
            raise ValueError("q_shape must contain positive dimensions") from exc
        object.__setattr__(self, "q_shape", shape)
        if not isinstance(self.nstates, int) or self.nstates < 1:
            raise ValueError("nstates must be a positive integer")
        if not self.q_shape or any(not isinstance(n, int) or n < 1 for n in self.q_shape):
            raise ValueError("q_shape must contain positive dimensions")
        if not self.basis_id:
            raise ValueError("basis_id must identify the electronic basis")
        if self.coordinate_kind not in {"cartesian", "normal_mode", "canonical"}:
            raise ValueError("only Cartesian or linear canonical coordinates are supported")

    @property
    def ndof(self) -> int:
        return prod(self.q_shape)
