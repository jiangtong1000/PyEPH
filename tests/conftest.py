"""Tests explicitly select scientific reference precision before creating arrays."""

import jax

jax.config.update("jax_enable_x64", True)
