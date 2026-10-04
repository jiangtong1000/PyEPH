"""Mergeable trajectory statistics, distinct from the legacy spread across ranks."""

from dataclasses import dataclass

import numpy as np


@dataclass
class EnsembleMoments:
    """Unweighted independent samples using the parallel Welford algorithm.

    For complex data variance is E[|x-mean|²]. This is not a covariance matrix of
    real/imaginary parts and not an autocorrelation-corrected time-series error.
    Samples must share the same physical time grid and observable definition.
    """

    count: int = 0
    mean: object = None
    m2: object = None

    def update(self, samples, axis=0):
        x = np.moveaxis(np.asarray(samples), axis, 0)
        if x.shape[0] == 0:
            return self
        if not np.isfinite(x).all():
            raise ValueError("cannot accumulate nonfinite samples")
        average = np.mean(x, axis=0)
        m2 = np.sum(np.abs(x - average)**2, axis=0)
        return self.merge(EnsembleMoments(x.shape[0], average, m2))

    def merge(self, other):
        if not other.count:
            return self
        if not self.count:
            self.count, self.mean, self.m2 = other.count, np.array(other.mean), np.array(other.m2)
            return self
        if np.shape(self.mean) != np.shape(other.mean):
            raise ValueError("cannot merge different observable shapes")
        count = self.count + other.count
        delta = other.mean - self.mean
        self.m2 = self.m2 + other.m2 + np.abs(delta)**2 * self.count * other.count / count
        self.mean = self.mean + delta * other.count / count
        self.count = count
        return self

    @property
    def variance(self):
        if self.count < 2:
            return np.full(np.shape(self.mean), np.nan)
        return self.m2 / (self.count - 1)

    @property
    def standard_error(self):
        return np.sqrt(self.variance / self.count) if self.count else np.nan
