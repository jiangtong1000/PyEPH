"""Compatibility wrapper for the archived Lanczos experiment.

The only active numerical implementation is pyeph.integrators.krylov. The
historical error_bound field is retained here for saved benchmark tooling; new
callers should use the production result's error_estimate name. Original scripts
and reports remain in a local historical archive, outside the distribution.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pyeph.integrators.krylov import (
    STATUS, LanczosError, LanczosOptions, require_success,
    lanczos_action as _checked_action,
)

# These names intentionally remain importable by the original benchmark tests.
__all__ = ["STATUS", "LanczosError", "LanczosOptions", "ActionResult",
           "lanczos_action", "require_success", "validation_report"]


class ActionResult(NamedTuple):
    value: Any
    error_bound: Any
    truncation_bound: Any
    recurrence_bound: Any
    roundoff_allowance: Any
    orthogonality_error: Any
    hermiticity_error: Any
    iterations: Any
    status: Any


def lanczos_action(apply, vectors, duration, options=LanczosOptions()):
    """Benchmark-only compatibility adapter; prefer the checked kernel directly."""
    return ActionResult(*_checked_action(apply, vectors, duration, options))


def validation_report():
    """Small independent NumPy/SciPy audit, not a speed benchmark."""
    import scipy.linalg

    rng = np.random.default_rng(276)
    records = []
    for size, dimension, duration in ((8, 8, .4), (32, 16, .4), (64, 24, .8), (32, 3, 2.0)):
        raw = rng.normal(size=(size, size)) + 1j*rng.normal(size=(size, size))
        matrix = (raw + raw.conj().T) / (2*np.sqrt(size))
        vector = rng.normal(size=size) + 1j*rng.normal(size=size)
        vector /= np.linalg.norm(vector)
        options = LanczosOptions(max_dimension=dimension)
        result = jax.jit(lambda h, v: lanczos_action(lambda x: h @ x, v, duration, options))(
            jnp.asarray(matrix), jnp.asarray(vector))
        reference = scipy.linalg.expm(-1j*duration*matrix) @ vector
        records.append(dict(size=size, capacity=dimension, duration=duration,
                            status=int(result.status), iterations=int(result.iterations),
                            error=float(np.linalg.norm(result.value-reference)),
                            error_bound=float(result.error_bound),
                            recurrence_bound=float(result.recurrence_bound),
                            roundoff_allowance=float(result.roundoff_allowance),
                            norm_error=float(abs(np.linalg.norm(result.value)-1))))
    blocks = []
    raw = rng.normal(size=(32, 32)) + 1j*rng.normal(size=(32, 32))
    matrix = (raw + raw.conj().T) / (2*np.sqrt(32))
    columns = rng.normal(size=(32, 3)) + 1j*rng.normal(size=(32, 3))
    columns /= np.linalg.norm(columns, axis=0)
    reference = scipy.linalg.expm(-.8j*matrix) @ columns
    for dimension in (16, 3):
        options = LanczosOptions(max_dimension=dimension)
        result = jax.jit(lambda h, v: lanczos_action(lambda x: h @ x, v, .8, options))(
            jnp.asarray(matrix), jnp.asarray(columns))
        value = np.asarray(result.value)
        blocks.append(dict(size=32, columns=3, capacity=dimension, duration=.8,
                           status=np.asarray(result.status).tolist(),
                           frobenius_error=float(np.linalg.norm(value-reference)),
                           frobenius_bound=float(np.linalg.norm(result.error_bound)),
                           gram_defect=float(np.linalg.norm(
                               value.conj().T @ value - columns.conj().T @ columns))))
    import platform
    from datetime import datetime, timezone
    from hashlib import sha256
    from pathlib import Path
    from pyeph.integrators import krylov
    return {"experimental": True, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "jax_version": jax.__version__, "numpy_version": np.__version__,
            "scipy_version": scipy.__version__, "platform": platform.platform(),
            "device": str(jax.devices()[0]), "x64": bool(jax.config.x64_enabled),
            "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "numerical_module_sha256": sha256(Path(krylov.__file__).read_bytes()).hexdigest(),
            "cases": records, "column_blocks": blocks}


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    report = json.dumps(validation_report(), indent=2, allow_nan=False)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(report + "\n")
    print(report)
