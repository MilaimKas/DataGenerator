"""Noise distribution generators for structural causal models."""

from __future__ import annotations

import numpy as np
from typing import Optional
from dataclasses import dataclass, field


class NoiseGenerator:
    """Base class for noise generators."""

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


@dataclass
class GaussianNoise(NoiseGenerator):
    """Gaussian (normal) noise."""
    mean: float = 0.0
    std: float = 1.0

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self.mean, self.std, n)


@dataclass
class UniformNoise(NoiseGenerator):
    """Uniform noise."""
    low: float = -1.0
    high: float = 1.0

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.low, self.high, n)


@dataclass
class LaplacianNoise(NoiseGenerator):
    """Laplacian (double exponential) noise - heavier tails than Gaussian."""
    loc: float = 0.0
    scale: float = 1.0

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.laplace(self.loc, self.scale, n)


@dataclass
class StudentTNoise(NoiseGenerator):
    """Student's t noise - controllable heavy tails."""
    df: float = 3.0  # degrees of freedom (lower = heavier tails)
    scale: float = 1.0

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_t(self.df, n) * self.scale


@dataclass
class MixtureNoise(NoiseGenerator):
    """Mixture of noise distributions."""
    components: list[NoiseGenerator] = field(default_factory=list)
    weights: Optional[list[float]] = None

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if not self.components:
            return np.zeros(n)
        weights = self.weights or [1.0 / len(self.components)] * len(self.components)
        weights = np.array(weights) / np.sum(weights)
        component_indices = rng.choice(len(self.components), size=n, p=weights)
        result = np.zeros(n)
        for i, comp in enumerate(self.components):
            mask = component_indices == i
            if mask.sum() > 0:
                result[mask] = comp.sample(mask.sum(), rng)
        return result
