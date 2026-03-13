"""Categorical variable support for DAG-based data generation.

Provides the CategoricalNodeInfo dataclass for storing metadata about
categorical variables expanded into one-hot sub-nodes, and a numerically
stable softmax utility.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MAX_CATEGORIES: int = 10


@dataclass
class CategoricalNodeInfo:
    """Metadata for a categorical variable expanded into one-hot sub-nodes.

    A categorical variable with K categories is internally represented as K
    binary sub-nodes in the DAG. This dataclass stores the mapping between
    the user-facing categorical name and its internal sub-node representation.

    Attributes:
        name: User-facing name of the categorical variable (e.g., "Color").
        categories: List of category labels (e.g., ["R", "G", "B"]).
        probabilities: Sampling probabilities for root nodes. None for child nodes
            whose distribution is determined by parents via softmax.
        sub_node_names: Internal sub-node names (e.g., ["Color_R", "Color_G", "Color_B"]).
    """

    name: str
    categories: list[str]
    probabilities: list[float] | None = None
    sub_node_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate and build sub-node names if not provided."""
        if len(self.categories) > MAX_CATEGORIES:
            raise ValueError(f"Too many categories ({len(self.categories)}). Maximum allowed is {MAX_CATEGORIES}.")
        if len(self.categories) < 2:
            raise ValueError("A categorical variable must have at least 2 categories.")
        if len(set(self.categories)) != len(self.categories):
            raise ValueError(f"Duplicate categories found in {self.categories}.")

        # Validate probabilities if provided (root node)
        if self.probabilities is not None:
            if len(self.probabilities) != len(self.categories):
                raise ValueError(
                    f"Length of probabilities ({len(self.probabilities)}) must match "
                    f"number of categories ({len(self.categories)})."
                )
            if any(p < 0 for p in self.probabilities):
                raise ValueError("Probabilities must be non-negative.")
            total = sum(self.probabilities)
            if not np.isclose(total, 1.0):
                raise ValueError(f"Probabilities must sum to 1.0, got {total:.6f}.")

        # Build sub-node names if not already set
        if not self.sub_node_names:
            self.sub_node_names = [f"{self.name}_{cat}" for cat in self.categories]


def softmax(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax with numerical stability.

    Args:
        scores: Array of shape (n, K) where K is the number of categories.

    Returns:
        Array of shape (n, K) with probabilities summing to 1 per row.
    """
    # Subtract row max for numerical stability (prevents overflow in exp)
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)
