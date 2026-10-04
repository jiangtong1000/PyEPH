# Candidate graph lifecycle

`pyeph.models.neighbors.NeighborGraph` builds immutable candidate graphs for
`LocalBlockModel`. It uses the existing atom-to-center map and graph contract;
it does not introduce a different Hamiltonian or dynamics method. Coordinates,
cell vectors, `switch_on`, `cutoff` and `skin` must use the same length unit.
The provider still declares the electronic basis, carrier convention, energy
units and nuclear reference potential.

```python
import numpy as np
from pyeph.models.local import AtomCenterMap, LocalBlockModel, LocalCoefficients
from pyeph.models.neighbors import NeighborGraph
import jax.numpy as jnp

centers = AtomCenterMap((0, 1, 2), (1.0, 1.0, 1.0), 3)
q = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
candidates = NeighborGraph(
    centers, q, norbitals=1, switch_on=2.5, cutoff=3.5, skin=1.0,
    capacity=20,
)

def coefficients(params, q_atoms, geometry):
    # Illustrative fixed orthonormal site model, not a material parameterization.
    # LocalBlockModel applies the physical smooth hopping support exactly once.
    return LocalCoefficients(jnp.zeros((3, 1, 1)),
                             jnp.exp(-geometry.distances)[:, None, None])

model = LocalBlockModel(candidates.graph, centers, coefficients)
candidates.require_coverage(q)
model.validate_at(None, jnp.asarray(q))
construction_record = candidates.metadata()
construction_id = candidates.identity
```

## Periodic images and wrapping

Cell vectors are **rows**. An edge `(a,b,nx,ny,nz)` has displacement
`R[b] + image @ cell - R[a]`. The graph stores each Hermitian pair once:
`a < b` permits every integer image; `a == b` permits only lexicographically
positive images. The zero-image self interaction belongs to the provider's
onsite block. A periodic graph represents a Gamma-point simulation supercell;
it does not insert Bloch phases.

Search includes **all** images within the exact sum of the float64 `cutoff`
and `skin` inputs, including self images
and multiple images of the same site pair. It does not assume an orthogonal
cell, a nearest image, or a cutoff smaller than half a cell length. Reciprocal
cell bounds use an exact rational 3×3 inverse and integer square-root bounds
to delimit a complete finite image search. Cartesian squared distances are
compared as exact integers after scaling the binary input geometry once to
a common power-of-two denominator. No rounded norm decides inclusion.
`max_image_checks` bounds total candidate-image work and raises before
constructing image pools or enumerating an oversized search. Construction is
a host pair search, with cost proportional
to site-pair count times the number of tested images; no linear-scaling or
accelerated neighbor-search performance is claimed.

`wrap_positions` returns `(wrapped, images)` with
`original = wrapped + images @ cell`. `unwrap_positions` requires those
explicit images. It never guesses how many times a particle crossed a cell.
`wrap_atoms` applies a common lattice shift to all atoms assigned to a center,
including zero-weight atoms, preserving fragment orientation and internal
coordinates. Its input fragments must already be coherently unwrapped; a
split molecule cannot be reconstructed without connectivity/winding input.
Translations that lose subcell coordinate or fragment geometry precision are
rejected. In particular, large integer images being representable does not
mean fractional coordinates survive adding their Cartesian translations. Use a
nearer coordinate origin rather than retaining arbitrarily large windings in
floating-point positions.

```python
from pyeph.models.neighbors import wrap_atoms, unwrap_atoms

periodic = NeighborGraph(centers, q, 1, 2.5, 3.5, 1.0,
                         cell=np.diag([10.0, 10.0, 10.0]), capacity=100)
wrapped, images = wrap_atoms(q, centers, periodic.cell)
wrapped_candidates = periodic.rewrapped(-images)
wrapped_candidates.require_coverage(wrapped, exhaustive=True)
restored = unwrap_atoms(wrapped, images, centers, periodic.cell)
np.testing.assert_allclose(restored, q)
```

`rewrapped` changes the reference coordinates and graph images together,
preserving candidate order and intended physical displacements. It returns a new
generation with parent identity only after a complete host search confirms that
the translated float64 reference geometry still has every required candidate.
Translation rounding can move an omitted pair across the list boundary; that
case rejects and requires an explicit rebuild and parameter remap. Verification
does not insert edges or change their existing order. Rewrapping only the
coordinates invalidates
the old graph convention and can remove a physically required image from the
represented set. Coordinate-dependent provider terms and reference potentials
must separately respect the intended periodic convention; shifting graph
images does not repair a nonperiodic provider.

## Coverage, capacity and explicit rebuilds

A snapshot contains every center pair within `cutoff + skin` at its reference
geometry. If each center has moved by at most `skin/2`, the triangle inequality
guarantees every current pair inside the physical cutoff is still present.
The fixed cell, fixed center map and consistent winding convention are part
of that statement. The check is conservative: uniform translation can exhaust
the skin even though pair distances are unchanged.

This is a **host geometric certificate**: coordinates, weights, cell, cutoff and
skin are interpreted as exact real numbers after float64 input conversion.
Weighted centers and squared-distance decisions use exact binary-rational or
integer arithmetic, including subnormal and very large inputs. The approximate
`maximum_displacement` display value need not reproduce the exact `within_skin`
decision when manually compared to a floating `skin/2` near a boundary. Zero skin
accepts an unchanged reference or exactly center-preserving motion.

The certificate does not bound independent native CPU/GPU arithmetic, provider
internal neighborhoods, or hidden dynamics stages. Native stage guarding needs
its own numerical margin and provider contract; see
[the proposed guard design](NEIGHBOR_GUARD_DESIGN.md). Exact host checks cost more
than ordinary floating distance checks and remain an explicitly bounded host
operation, outside compiled propagation. No fast neighbor-update claim follows.

`check(q)` returns displacement and skin evidence. Without an exhaustive search,
`covered=True` means the skin proves current coverage; `covered=None` means the
certificate expired. `check(q, exhaustive=True)` searches
all current physical pairs and reports the exact missing edge tuples, or an
empty tuple and `covered=True` if this geometry is covered, even if the skin
has expired. An expired skin and an omitted edge
are different diagnostics. `require_coverage` raises `NeighborCoverageError`
when the skin expires, retaining the report even if exhaustive search happens
to find no missing pair at that instant.

`capacity` is a hard limit on actual unique edges, not a padding size. A
`NeighborCapacityError` retains both `required` and `capacity`; the old snapshot
is unchanged. Raise the budget explicitly and retry. No edge is truncated and
no dummy edge enters the Hamiltonian.

```python
q_next = q.copy()
q_next[2, 0] -= 0.7
report = candidates.check(q_next, exhaustive=True)
if report.rebuild_required:
    replacement = candidates.rebuild(q_next, capacity=30)
    replacement_model = LocalBlockModel(replacement.graph, centers, coefficients)
    replacement_model.validate_at(None, jnp.asarray(q_next))
    assert replacement.parent_identity == candidates.identity
```

Each generation has fixed array shapes during differentiation/JIT. A rebuild
can change the edge count, so construct a new model and execution object and
save the new generation's metadata in the campaign manifest. Per-edge parameter
arrays must be remapped by the full `(a,b,image)` key; merely reusing the old
array order is unsafe. Provider weights and data retain their own provenance.
Normal model manifests already identify graph edges and cutoff; the separate
construction record adds reference coordinates, budgets and parent lineage.

**No automatic Runner guard is installed.** A check covers the supplied
geometry, not hidden intermediate electronic/nuclear stages or the path between
saved frames. A workflow must validate all required evaluation geometries or
supply a proven displacement bound for the complete compiled interval. If an
interval crosses the validity bound, roll back to a valid checkpoint, rebuild
and repeat it; do not silently continue from dynamics computed with omitted
interactions. Cell deformation also requires a freshly constructed snapshot.
The proposed [stage guard and recovery workflow](NEIGHBOR_GUARD_DESIGN.md)
specifies the next extension; its interface is not implemented yet.

## Smoothness and evidence

Graph rebuilding is a discrete host operation. Differentiation acts on the
smooth provider and `LocalBlockModel`'s quintic cutoff with a fixed candidate
set. Inserting/removing a candidate outside physical support must not change
the model. This requires **all** provider graph dependencies to vanish smoothly
at their declared support, including onsite neighbor messages, descriptor
normalization and baseline terms. The graph's final hopping switch alone does
not guarantee that property for an arbitrary provider.

The focused tests compare orthogonal and strongly skew periodic cells against
an independent oversized image cube; test two-endpoint skin motion and omitted
pairs; retain failures on capacity exhaustion; round-trip explicit multi-cell
windings while preserving fragment geometry; and compare complete cutoff
forces with independent finite differences. Candidate insertion/removal outside
support preserves energy and force in the tested radial provider. These tests
establish geometry and numerical contracts, not material-model accuracy,
variable-cell dynamics, or production neighbor-search scaling.
