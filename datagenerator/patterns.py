"""Utility functions for creating common DAG patterns."""

from __future__ import annotations

import numpy as np
from typing import Optional, Union

from .dag import DAG
from .transforms import Transform, get_transform, transforms_available


def create_chain(
    n_nodes: int,
    weights: Optional[list[float]] = None,
    transforms: Optional[list[Union[str, Transform]]] = None,
    noise_std: float = 1.0,
    names: Optional[list[str]] = None,
) -> DAG:
    """
    Create a chain DAG: X1 -> X2 -> ... -> Xn

    Args:
        n_nodes: Number of nodes in the chain
        weights: Edge weights (length n_nodes - 1)
        transforms: Edge transforms (length n_nodes - 1)
        noise_std: Noise standard deviation for all nodes
        names: Custom node names

    Returns:
        DAG with chain structure
    """
    dag = DAG()
    names = names or [f"X{i}" for i in range(n_nodes)]
    weights = weights or [1.0] * (n_nodes - 1)
    transforms = transforms or ["linear"] * (n_nodes - 1)

    for name in names:
        dag.add_node(name, noise_std=noise_std)

    for i in range(n_nodes - 1):
        dag.add_edge(names[i], names[i + 1], weight=weights[i], transform=transforms[i])

    return dag


def create_fork(
    n_children: int = 2,
    confounder_name: str = "Z",
    child_names: Optional[list[str]] = None,
    weights: Optional[list[float]] = None,
    transforms: Optional[list[Union[str, Transform]]] = None,
    noise_std: float = 1.0,
) -> DAG:
    """
    Create a fork DAG (common cause): Z -> X1, Z -> X2, ..., Z -> Xn

    This creates confounding between the children.

    Args:
        n_children: Number of child nodes
        confounder_name: Name of the confounder node
        child_names: Names for child nodes
        weights: Edge weights
        transforms: Edge transforms
        noise_std: Noise standard deviation

    Returns:
        DAG with fork structure
    """
    dag = DAG()
    child_names = child_names or [f"X{i}" for i in range(n_children)]
    weights = weights or [1.0] * n_children
    transforms = transforms or ["linear"] * n_children

    dag.add_node(confounder_name, noise_std=noise_std)
    for i, name in enumerate(child_names):
        dag.add_node(name, noise_std=noise_std)
        dag.add_edge(confounder_name, name, weight=weights[i], transform=transforms[i])

    return dag


def create_collider(
    n_parents: int = 2,
    collider_name: str = "Y",
    parent_names: Optional[list[str]] = None,
    weights: Optional[list[float]] = None,
    transforms: Optional[list[Union[str, Transform]]] = None,
    noise_std: float = 1.0,
) -> DAG:
    """
    Create a collider DAG: X1 -> Y, X2 -> Y, ..., Xn -> Y

    Parents are marginally independent but conditionally dependent given Y.

    Args:
        n_parents: Number of parent nodes
        collider_name: Name of the collider node
        parent_names: Names for parent nodes
        weights: Edge weights
        transforms: Edge transforms
        noise_std: Noise standard deviation

    Returns:
        DAG with collider structure
    """
    dag = DAG()
    parent_names = parent_names or [f"X{i}" for i in range(n_parents)]
    weights = weights or [1.0] * n_parents
    transforms = transforms or ["linear"] * n_parents

    for name in parent_names:
        dag.add_node(name, noise_std=noise_std)
    dag.add_node(collider_name, noise_std=noise_std)

    for i, name in enumerate(parent_names):
        dag.add_edge(name, collider_name, weight=weights[i], transform=transforms[i])

    return dag


def create_mediator(
    treatment_name: str = "X",
    mediator_name: str = "M",
    outcome_name: str = "Y",
    direct_effect: float = 1.0,
    indirect_effect_xm: float = 1.0,
    indirect_effect_my: float = 1.0,
    transforms: Optional[dict[str, Union[str, Transform]]] = None,
    noise_std: float = 1.0,
) -> DAG:
    """
    Create a mediation DAG: X -> M -> Y and X -> Y

    Allows studying direct vs indirect effects.

    Args:
        treatment_name: Name of treatment variable
        mediator_name: Name of mediator variable
        outcome_name: Name of outcome variable
        direct_effect: Weight for X -> Y (direct path)
        indirect_effect_xm: Weight for X -> M
        indirect_effect_my: Weight for M -> Y
        transforms: Dict of transforms for each edge
        noise_std: Noise standard deviation

    Returns:
        DAG with mediation structure
    """
    dag = DAG()
    transforms = transforms or {}

    dag.add_node(treatment_name, noise_std=noise_std)
    dag.add_node(mediator_name, noise_std=noise_std)
    dag.add_node(outcome_name, noise_std=noise_std)

    dag.add_edge(treatment_name, mediator_name, weight=indirect_effect_xm,
                transform=transforms.get("X->M", "linear"))
    dag.add_edge(mediator_name, outcome_name, weight=indirect_effect_my,
                transform=transforms.get("M->Y", "linear"))
    dag.add_edge(treatment_name, outcome_name, weight=direct_effect,
                transform=transforms.get("X->Y", "linear"))

    return dag


def create_instrument(
    instrument_name: str = "Z",
    treatment_name: str = "X",
    outcome_name: str = "Y",
    confounder_name: str = "U",
    z_x_weight: float = 1.0,
    x_y_weight: float = 1.0,
    u_x_weight: float = 0.5,
    u_y_weight: float = 0.5,
    transforms: Optional[dict[str, Union[str, Transform]]] = None,
    noise_std: float = 1.0,
) -> DAG:
    """
    Create an instrumental variable DAG: Z -> X -> Y with U -> X and U -> Y

    Z is an instrument (affects Y only through X), U is a confounder.

    Args:
        instrument_name: Name of instrument variable
        treatment_name: Name of treatment variable
        outcome_name: Name of outcome variable
        confounder_name: Name of confounder variable
        z_x_weight: Weight for Z -> X
        x_y_weight: Weight for X -> Y (causal effect of interest)
        u_x_weight: Weight for U -> X (confounding)
        u_y_weight: Weight for U -> Y (confounding)
        transforms: Dict of transforms for each edge
        noise_std: Noise standard deviation

    Returns:
        DAG with instrumental variable structure
    """
    dag = DAG()
    transforms = transforms or {}

    dag.add_node(instrument_name, noise_std=noise_std)
    dag.add_node(confounder_name, noise_std=noise_std)
    dag.add_node(treatment_name, noise_std=noise_std)
    dag.add_node(outcome_name, noise_std=noise_std)

    dag.add_edge(instrument_name, treatment_name, weight=z_x_weight,
                transform=transforms.get("Z->X", "linear"))
    dag.add_edge(treatment_name, outcome_name, weight=x_y_weight,
                transform=transforms.get("X->Y", "linear"))
    dag.add_edge(confounder_name, treatment_name, weight=u_x_weight,
                transform=transforms.get("U->X", "linear"))
    dag.add_edge(confounder_name, outcome_name, weight=u_y_weight,
                transform=transforms.get("U->Y", "linear"))

    return dag


def create_random_dag(
    n_nodes: int,
    edge_probability: float = 0.3,
    weight_range: tuple[float, float] = (-1.0, 1.0),
    noise_std: float = 1.0,
    seed: Optional[int] = None,
    node_names: Optional[list[str]] = None,
    use_transform: Optional[bool] = False
) -> DAG:
    """
    Create a random DAG with specified number of nodes and edge probability.

    Args:
        n_nodes: Number of nodes
        edge_probability: Probability of edge between any ordered pair
        weight_range: Range for random edge weights
        noise_std: Noise standard deviation for all nodes
        seed: Random seed
        node_names: Custom node names
        use_transform: True if sample from transform function available list.

    Returns:
        Random DAG
    """
    rng = np.random.default_rng(seed)
    dag = DAG()

    names = node_names or [f"X{i}" for i in range(n_nodes)]

    for name in names:
        dag.add_node(name, noise_std=noise_std)

    # default
    transform_fct = "linear"

    # Only allow edges from earlier to later nodes (ensures acyclicity)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_probability:
                weight = rng.uniform(weight_range[0], weight_range[1])
                if use_transform:
                    transform_fct = np.random.choice(transforms_available)
                dag.add_edge(names[i], names[j], weight=weight, transform=get_transform(transform_fct))
    return dag
