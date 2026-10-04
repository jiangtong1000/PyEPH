"""Explicit numerical policy for the historical NumPy64 compatibility API."""


def require_legacy_precision():
    import jax

    if not jax.config.x64_enabled:
        raise ValueError(
            "The legacy GreenKubo facade requires 64-bit JAX to preserve its NumPy64 "
            "Hamiltonians and initial samples. Set JAX_ENABLE_X64=true before starting "
            "Python, or call pyeph.configure_precision() before constructing the simulation."
        )
