"""Non-linear transformations for causal edge relationships."""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass, field


class Transform:
    """Base class for transformations applied to parent contributions."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def math_notation(self, var: str) -> str:
        """Return mathematical notation for this transform applied to var."""
        return f"{self.__class__.__name__}({var})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class IdentityTransform(Transform):
    """No transformation (linear)."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x

    def math_notation(self, var: str) -> str:
        return var


@dataclass
class PolynomialTransform(Transform):
    """Polynomial transformation: sum of x^k for k in degrees."""
    degrees: list[int] = field(default_factory=lambda: [1])
    coefficients: Optional[list[float]] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        coeffs = self.coefficients or [1.0] * len(self.degrees)
        result = np.zeros_like(x)
        for deg, coef in zip(self.degrees, coeffs):
            result += coef * np.power(x, deg)
        return result

    def math_notation(self, var: str) -> str:
        coeffs = self.coefficients or [1.0] * len(self.degrees)
        terms = []
        for deg, coef in zip(self.degrees, coeffs):
            if deg == 1 and coef == 1.0:
                terms.append(var)
            elif deg == 1:
                terms.append(f"{coef:.2f} * {var}")
            elif coef == 1.0:
                terms.append(f"{var}^{deg}")
            else:
                terms.append(f"{coef:.2f} * {var}^{deg}")
        if len(terms) == 1:
            return terms[0]
        return "(" + " + ".join(terms) + ")"

    def __repr__(self) -> str:
        return f"PolynomialTransform(degrees={self.degrees})"


@dataclass
class SigmoidTransform(Transform):
    """Sigmoid transformation: 1 / (1 + exp(-scale * x))."""
    scale: float = 1.0
    shift: float = 0.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.scale * (x - self.shift)))

    def math_notation(self, var: str) -> str:
        if self.scale == 1.0 and self.shift == 0.0:
            return f"sigmoid({var})"
        elif self.shift == 0.0:
            return f"sigmoid({self.scale:.2f} * {var})"
        else:
            return f"sigmoid({self.scale:.2f} * ({var} - {self.shift:.2f}))"

    def __repr__(self) -> str:
        return f"SigmoidTransform(scale={self.scale})"


@dataclass
class TanhTransform(Transform):
    """Hyperbolic tangent transformation."""
    scale: float = 1.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(self.scale * x)

    def math_notation(self, var: str) -> str:
        if self.scale == 1.0:
            return f"tanh({var})"
        return f"tanh({self.scale:.2f} * {var})"

    def __repr__(self) -> str:
        return f"TanhTransform(scale={self.scale})"


@dataclass
class SinusoidalTransform(Transform):
    """Sinusoidal transformation: amplitude * sin(frequency * x + phase)."""
    amplitude: float = 1.0
    frequency: float = 1.0
    phase: float = 0.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.amplitude * np.sin(self.frequency * x + self.phase)

    def math_notation(self, var: str) -> str:
        # Build inner: frequency * var + phase
        if self.frequency == 1.0:
            inner = var
        else:
            inner = f"{self.frequency:.2f} * {var}"
        if self.phase != 0.0:
            inner = f"{inner} + {self.phase:.2f}"
        # Build outer: amplitude * sin(inner)
        if self.amplitude == 1.0:
            return f"sin({inner})"
        return f"{self.amplitude:.2f} * sin({inner})"

    def __repr__(self) -> str:
        return f"SinusoidalTransform(freq={self.frequency})"


@dataclass
class ExponentialTransform(Transform):
    """Exponential transformation: scale * exp(rate * x)."""
    scale: float = 1.0
    rate: float = 1.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Clip to avoid overflow
        clipped = np.clip(self.rate * x, -50, 50)
        return self.scale * np.exp(clipped)

    def math_notation(self, var: str) -> str:
        if self.rate == 1.0:
            inner = var
        else:
            inner = f"{self.rate:.2f} * {var}"
        if self.scale == 1.0:
            return f"exp({inner})"
        return f"{self.scale:.2f} * exp({inner})"

    def __repr__(self) -> str:
        return f"ExponentialTransform(rate={self.rate})"


@dataclass
class LogTransform(Transform):
    """Log transformation: scale * log(|x| + epsilon)."""
    scale: float = 1.0
    epsilon: float = 1e-6

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.scale * np.log(np.abs(x) + self.epsilon)

    def math_notation(self, var: str) -> str:
        if self.scale == 1.0:
            return f"log(|{var}|)"
        return f"{self.scale:.2f} * log(|{var}|)"

    def __repr__(self) -> str:
        return f"LogTransform(scale={self.scale})"


@dataclass
class ReLUTransform(Transform):
    """ReLU transformation: max(0, x) or leaky variant."""
    negative_slope: float = 0.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.negative_slope * x)

    def math_notation(self, var: str) -> str:
        if self.negative_slope == 0.0:
            return f"max(0, {var})"
        return f"leaky_relu({var}, α={self.negative_slope:.2f})"

    def __repr__(self) -> str:
        return f"ReLUTransform(negative_slope={self.negative_slope})"


@dataclass
class ThresholdTransform(Transform):
    """Threshold/step transformation."""
    threshold: float = 0.0
    below_value: float = 0.0
    above_value: float = 1.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > self.threshold, self.above_value, self.below_value)

    def math_notation(self, var: str) -> str:
        return f"I({var} > {self.threshold:.2f})"

    def __repr__(self) -> str:
        return f"ThresholdTransform(threshold={self.threshold})"


@dataclass
class CompositeTransform(Transform):
    """Compose multiple transformations: t_n(...t_2(t_1(x)))."""
    transforms: list[Transform] = field(default_factory=list)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        result = x
        for t in self.transforms:
            result = t(result)
        return result

    def math_notation(self, var: str) -> str:
        result = var
        for t in self.transforms:
            result = t.math_notation(result)
        return result

    def __repr__(self) -> str:
        return f"CompositeTransform({self.transforms})"


@dataclass
class CustomTransform(Transform):
    """Custom transformation using a user-provided function."""
    func: Callable[[np.ndarray], np.ndarray] = field(default=lambda x: x)
    name: str = "custom"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.func(x)

    def math_notation(self, var: str) -> str:
        return f"{self.name}({var})"

    def __repr__(self) -> str:
        return f"CustomTransform({self.name})"


def get_transform(name: str, **kwargs) -> Transform:
    """Get a transform by name with optional parameters."""
    transforms = {
        "linear": IdentityTransform,
        "identity": IdentityTransform,
        "quadratic": lambda: PolynomialTransform(degrees=[2]),
        "cubic": lambda: PolynomialTransform(degrees=[3]),
        "polynomial": lambda: PolynomialTransform(**kwargs),
        "sigmoid": lambda: SigmoidTransform(**kwargs),
        "tanh": lambda: TanhTransform(**kwargs),
        "sin": lambda: SinusoidalTransform(**kwargs),
        "sinusoidal": lambda: SinusoidalTransform(**kwargs),
        "exp": lambda: ExponentialTransform(**kwargs),
        "exponential": lambda: ExponentialTransform(**kwargs),
        "log": lambda: LogTransform(**kwargs),
        "relu": lambda: ReLUTransform(**kwargs),
        "leaky_relu": lambda: ReLUTransform(negative_slope=0.1, **kwargs),
        "threshold": lambda: ThresholdTransform(**kwargs),
        "step": lambda: ThresholdTransform(**kwargs),
    }
    if name not in transforms:
        raise ValueError(f"Unknown transform: {name}. Available: {list(transforms.keys())}")
    return transforms[name]() if callable(transforms[name]) else transforms[name]
