"""Data generator for sampling from DAG structural equations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .categorical import softmax
from .dag import DAG


class DataGenerator:
    """Generate synthetic data from a DAG using structural equations.

    For each node X with parents Pa(X), the structural equation is:
        X = sum_{P in Pa(X)} [weight_P * transform_P(P)] + noise_X

    Categorical variables are handled as groups of one-hot sub-nodes.
    Root categoricals are sampled via multinomial; child categoricals
    use softmax over per-category scores computed from parent contributions.
    """

    def __init__(self, dag: DAG, seed: int | None = None):
        """Initialize the data generator.

        Args:
            dag: The DAG defining the causal structure.
            seed: Random seed for reproducibility.
        """
        self.dag = dag
        self.rng = np.random.default_rng(seed)

    def sample(
        self,
        n: int,
        return_dict: bool = False,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Sample n data points from the DAG.

        Args:
            n: Number of samples.
            return_dict: If True, return dict with node names as keys.

        Returns:
            Either a 2D numpy array (n x num_nodes) or a dict.
        """
        order = self.dag._compute_topological_order()
        data: dict[str, np.ndarray] = {}
        processed_categoricals: set[str] = set()

        for name in order:
            # Check if this is a sub-node of a categorical variable
            cat_name = self.dag.get_categorical_owner(name)
            if cat_name is not None:
                if cat_name in processed_categoricals:
                    continue  # already processed this group
                processed_categoricals.add(cat_name)
                self._sample_categorical(cat_name, n, data)
                continue

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

    def _sample_categorical(
        self,
        cat_name: str,
        n: int,
        data: dict[str, np.ndarray],
    ) -> None:
        """Sample a categorical variable and set its one-hot sub-node values.

        Args:
            cat_name: Name of the categorical variable.
            n: Number of samples.
            data: Shared data dict to write sub-node arrays into.
        """
        info = self.dag.categorical_nodes[cat_name]

        # Handle interventions: all sub-nodes will be intervened together
        if any(self.dag.nodes[sn].intervened for sn in info.sub_node_names):
            for sub_name in info.sub_node_names:
                data[sub_name] = np.full(n, self.dag.nodes[sub_name].intervention_value)
            return

        # Check if root (no incoming edges on any sub-node)
        is_root = all(len(self.dag.edges.get(sn, [])) == 0 for sn in info.sub_node_names)

        if is_root:
            # Multinomial sampling from user-defined probabilities
            probs = info.probabilities
            if probs is None:
                raise ValueError(f"Root categorical '{cat_name}' has no probabilities defined.")
            choices = self.rng.choice(len(info.categories), size=n, p=probs)
        else:
            # Child categorical: compute latent scores per category, then softmax
            k = len(info.categories)
            scores = np.zeros((n, k))
            for idx, sub_name in enumerate(info.sub_node_names):
                for edge in self.dag.edges.get(sub_name, []):
                    parent_value = data[edge.source]
                    scores[:, idx] += edge.compute_contribution(parent_value)

            # Softmax → probabilities, then sample via vectorised cumulative-sum trick
            probs_matrix = softmax(scores)
            cumprobs = np.cumsum(probs_matrix, axis=1)
            u = self.rng.random(n)[:, np.newaxis]
            choices = (u > cumprobs).sum(axis=1)

        # Set one-hot sub-node values
        for idx, sub_name in enumerate(info.sub_node_names):
            data[sub_name] = (choices == idx).astype(float)

    def sample_interventional(
        self,
        n: int,
        interventions: dict[str, float | str],
        return_dict: bool = False,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Sample with temporary interventions (doesn't modify the DAG).

        For categorical nodes, pass the category name as the value
        (e.g., ``{"Color": "R"}``).

        Args:
            n: Number of samples.
            interventions: Dict of {node_name: intervention_value}.
            return_dict: If True, return dict with node names as keys.

        Returns:
            Sampled data with interventions applied.
        """
        # Save current intervention state and apply interventions
        saved_state: dict[str, list[tuple[str, bool, float | None]]] = {}
        for name, value in interventions.items():
            if self.dag.is_categorical(name):
                info = self.dag.categorical_nodes[name]
                saved_state[name] = [
                    (sn, self.dag.nodes[sn].intervened, self.dag.nodes[sn].intervention_value)
                    for sn in info.sub_node_names
                ]
                self.dag.intervene(name, value)
            elif name in self.dag.nodes:
                if isinstance(value, str):
                    raise TypeError(
                        f"String value '{value}' is only valid for categorical nodes, "
                        f"but '{name}' is continuous."
                    )
                node = self.dag.nodes[name]
                saved_state[name] = [(name, node.intervened, node.intervention_value)]
                node.intervened = True
                node.intervention_value = value
            else:
                raise ValueError(f"Node '{name}' not in DAG")

        try:
            return self.sample(n, return_dict=return_dict)
        finally:
            # Restore original state
            for entries in saved_state.values():
                for sub_name, was_intervened, old_value in entries:
                    self.dag.nodes[sub_name].intervened = was_intervened
                    self.dag.nodes[sub_name].intervention_value = old_value

    def show_equations(self) -> str:
        """Return the structural equations in mathematical notation."""
        return self.dag.show_equations()

    def get_column_names(self) -> list[str]:
        """Get column names in the order they appear in sample output."""
        return self.dag._get_user_facing_order()

    def to_dataframe(self, n: int) -> pd.DataFrame:
        """Sample and return as a pandas DataFrame.

        Categorical variables are collapsed from one-hot sub-columns
        into a single ``pd.Categorical`` column.
        """
        data = self.sample(n, return_dict=True)

        columns: dict[str, np.ndarray | pd.Categorical] = {}
        seen_categoricals: set[str] = set()

        for name in self.dag._compute_topological_order():
            cat_name = self.dag.get_categorical_owner(name)
            if cat_name is not None:
                if cat_name not in seen_categoricals:
                    seen_categoricals.add(cat_name)
                    info = self.dag.categorical_nodes[cat_name]
                    # Reconstruct category labels from one-hot encoding
                    one_hot = np.column_stack([data[sn] for sn in info.sub_node_names])
                    indices = np.argmax(one_hot, axis=1)
                    columns[cat_name] = pd.Categorical(
                        [info.categories[i] for i in indices],
                        categories=info.categories,
                    )
            else:
                columns[name] = data[name]

        return pd.DataFrame(columns)
