"""Data generator for sampling from DAG structural equations."""

from __future__ import annotations

import numpy as np
from typing import Optional, Union

from .dag import DAG


class DataGenerator:
    """
    Generate synthetic data from a DAG using structural equations.

    For each node X with parents Pa(X), the structural equation is:
        X = sum_{P in Pa(X)} [weight_P * transform_P(P)] + noise_X
    """

    def __init__(self, dag: DAG, seed: Optional[int] = None):
        """
        Initialize the data generator.

        Args:
            dag: The DAG defining the causal structure
            seed: Random seed for reproducibility
        """
        self.dag = dag
        self.rng = np.random.default_rng(seed)

    def sample(
        self,
        n: int,
        return_dict: bool = False,
    ) -> Union[np.ndarray, dict[str, np.ndarray]]:
        """
        Sample n data points from the DAG.

        Args:
            n: Number of samples
            return_dict: If True, return dict with node names as keys

        Returns:
            Either a 2D numpy array (n x num_nodes) or a dict
        """
        order = self.dag._compute_topological_order()
        data = {}

        for name in order:
            node = self.dag.nodes[name]

            if node.intervened:
                # Intervention: fixed value
                data[name] = np.full(n, node.intervention_value)
            else:
                # Structural equation: sum of parent contributions + noise
                value = np.zeros(n)

                for edge in self.dag.edges.get(name, []):
                    parent_value = data[edge.source]
                    value += edge.compute_contribution(parent_value)

                # Add noise
                value += node.noise.sample(n, self.rng)
                data[name] = value

        if return_dict:
            return data
        else:
            # Return as 2D array in topological order
            return np.column_stack([data[name] for name in order])

    def sample_interventional(
        self,
        n: int,
        interventions: dict[str, float],
        return_dict: bool = False,
    ) -> Union[np.ndarray, dict[str, np.ndarray]]:
        """
        Sample with temporary interventions (doesn't modify the DAG).

        Args:
            n: Number of samples
            interventions: Dict of {node_name: intervention_value}
            return_dict: If True, return dict with node names as keys

        Returns:
            Sampled data with interventions applied
        """
        # Save current intervention state
        saved_state = {}
        for name in interventions:
            if name not in self.dag.nodes:
                raise ValueError(f"Node '{name}' not in DAG")
            node = self.dag.nodes[name]
            saved_state[name] = (node.intervened, node.intervention_value)
            node.intervened = True
            node.intervention_value = interventions[name]

        try:
            return self.sample(n, return_dict=return_dict)
        finally:
            # Restore state
            for name, (intervened, value) in saved_state.items():
                node = self.dag.nodes[name]
                node.intervened = intervened
                node.intervention_value = value

    def show_equations(self) -> str:
        """Return the structural equations in mathematical notation."""
        return self.dag.show_equations()

    def get_column_names(self) -> list[str]:
        """Get column names in the order they appear in sample output."""
        return self.dag._compute_topological_order()

    def to_dataframe(self, n: int):
        """Sample and return as a pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")

        data = self.sample(n, return_dict=True)
        return pd.DataFrame(data)
