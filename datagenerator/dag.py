"""Directed Acyclic Graph for structural causal models."""

from __future__ import annotations

import numpy as np
from typing import Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

from .noise import NoiseGenerator, GaussianNoise, UniformNoise, LaplacianNoise, StudentTNoise
from .transforms import Transform, IdentityTransform, get_transform


@dataclass
class Edge:
    """Represents an edge in the DAG with associated weight and transformation."""
    source: str
    target: str
    weight: float = 1.0
    transform: Transform = field(default_factory=IdentityTransform)

    def compute_contribution(self, parent_value: np.ndarray) -> np.ndarray:
        """Compute this edge's contribution to the target node."""
        return self.weight * self.transform(parent_value)


@dataclass
class Node:
    """
    Represents a node in the DAG with its structural equation.

    The structural equation for node X is:
        X = f(parents) + noise
    where f(parents) = sum of edge contributions from all parents
    """
    name: str
    noise: NoiseGenerator = field(default_factory=lambda: GaussianNoise(std=1.0))
    is_root: bool = True  # Updated when edges are added

    # For interventions (do-calculus)
    intervened: bool = False
    intervention_value: Optional[float] = None


class DAG:
    """
    Directed Acyclic Graph for structural causal models.

    Supports:
    - Adding nodes with custom noise distributions
    - Adding edges with weights and non-linear transformations
    - Cycle detection
    - Topological ordering for sampling
    - Interventions (do-calculus)
    - Visualization of the graph structure
    """

    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, list[Edge]] = defaultdict(list)  # target -> list of incoming edges
        self._topological_order: Optional[list[str]] = None

    def add_node(
        self,
        name: str,
        noise: Optional[NoiseGenerator] = None,
        noise_type: str = "gaussian",
        noise_std: float = 1.0,
        noise_params: Optional[dict] = None,
    ) -> "DAG":
        """
        Add a node to the DAG.

        Args:
            name: Unique identifier for the node
            noise: NoiseGenerator instance (takes precedence if provided)
            noise_type: Type of noise ("gaussian", "uniform", "laplacian", "student_t")
            noise_std: Standard deviation for Gaussian noise (convenience parameter)
            noise_params: Additional parameters for noise generator

        Returns:
            Self for method chaining
        """
        if name in self.nodes:
            warnings.warn(f"Node '{name}' already exists, updating it.")

        if noise is None:
            noise_params = noise_params or {}
            if noise_type == "gaussian":
                noise = GaussianNoise(std=noise_std, **noise_params)
            elif noise_type == "uniform":
                noise = UniformNoise(**noise_params)
            elif noise_type == "laplacian":
                noise = LaplacianNoise(scale=noise_std, **noise_params)
            elif noise_type == "student_t":
                noise = StudentTNoise(scale=noise_std, **noise_params)
            else:
                raise ValueError(f"Unknown noise type: {noise_type}")

        self.nodes[name] = Node(name=name, noise=noise)
        self._topological_order = None  # Invalidate cache
        return self

    def add_edge(
        self,
        source: str,
        target: str,
        weight: float = 1.0,
        transform: Optional[Union[Transform, str]] = None,
        transform_params: Optional[dict] = None,
    ) -> "DAG":
        """
        Add a directed edge from source to target.

        Args:
            source: Name of the parent node
            target: Name of the child node
            weight: Linear weight/coefficient for this edge
            transform: Transform instance or name (e.g., "quadratic", "sigmoid")
            transform_params: Parameters for named transforms

        Returns:
            Self for method chaining
        """
        # Auto-create nodes if they don't exist
        if source not in self.nodes:
            self.add_node(source)
        if target not in self.nodes:
            self.add_node(target)

        # Mark target as non-root
        self.nodes[target].is_root = False

        # Handle transform
        if transform is None:
            transform = IdentityTransform()
        elif isinstance(transform, str):
            transform_params = transform_params or {}
            transform = get_transform(transform, **transform_params)

        edge = Edge(source=source, target=target, weight=weight, transform=transform)
        self.edges[target].append(edge)

        # Check for cycles
        self._topological_order = None
        try:
            self._compute_topological_order()
        except ValueError as e:
            # Remove the edge if it creates a cycle
            self.edges[target].pop()
            raise ValueError(f"Adding edge {source} -> {target} would create a cycle") from e

        return self

    def _compute_topological_order(self) -> list[str]:
        """
            Compute topological ordering using Kahn's algorithm.
            It:
                - computes the order in which nodes must be processed (parents before children). Needed for sampling.
                - detects cycles in the graph.
                - caches the result for future calls.
        """
        if self._topological_order is not None:
            return self._topological_order

        # Compute in-degrees
        in_degree = {name: 0 for name in self.nodes}
        for target, edges in self.edges.items():
            in_degree[target] = len(edges)

        # Start with root nodes (in-degree 0)
        queue = [name for name, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)

            # Find all children of this node
            for target, edges in self.edges.items():
                for edge in edges:
                    if edge.source == node:
                        in_degree[target] -= 1
                        if in_degree[target] == 0:
                            queue.append(target)

        if len(order) != len(self.nodes):
            raise ValueError("Graph contains a cycle!")

        self._topological_order = order
        return order

    def get_parents(self, node: str) -> list[str]:
        """Get parent nodes of a given node."""
        return [edge.source for edge in self.edges.get(node, [])]

    def get_children(self, node: str) -> list[str]:
        """Get child nodes of a given node."""
        children = []
        for target, edges in self.edges.items():
            for edge in edges:
                if edge.source == node:
                    children.append(target)
        return children

    def intervene(self, node: str, value: float) -> "DAG":
        """
        Set an intervention on a node (do-calculus).

        This fixes the node to a constant value, breaking the influence
        of its parents.

        Args:
            node: Name of the node to intervene on
            value: Fixed value for the intervention

        Returns:
            Self for method chaining
        """
        if node not in self.nodes:
            raise ValueError(f"Node '{node}' not in DAG")
        self.nodes[node].intervened = True
        self.nodes[node].intervention_value = value
        return self

    def clear_interventions(self) -> "DAG":
        """Clear all interventions."""
        for node in self.nodes.values():
            node.intervened = False
            node.intervention_value = None
        return self

    def copy(self) -> "DAG":
        """Create a deep copy of this DAG."""
        import copy
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        n_nodes = len(self.nodes)
        n_edges = sum(len(e) for e in self.edges.values())
        return f"DAG(nodes={n_nodes}, edges={n_edges})"

    def describe(self) -> str:
        """Return a detailed description of the DAG structure."""
        lines = ["DAG Structure:", "=" * 40]

        order = self._compute_topological_order()
        for name in order:
            node = self.nodes[name]
            parents = self.get_parents(name)

            if parents:
                parent_str = ", ".join(parents)
                lines.append(f"\n{name} <- {parent_str}")
                for edge in self.edges[name]:
                    lines.append(f"  {edge.source}: weight={edge.weight}, transform={edge.transform}")
            else:
                lines.append(f"\n{name} (root)")

            lines.append(f"  noise: {node.noise}")
            if node.intervened:
                lines.append(f"  INTERVENED: {node.intervention_value}")

        return "\n".join(lines)

    def plot(self, figsize: tuple[float, float] = (8, 6), show_weights: bool = True):
        """
        Plot the DAG using matplotlib.

        Args:
            figsize: Figure size (width, height)
            show_weights: Whether to display edge weights on the plot

        Returns:
            matplotlib figure and axes
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
        except ImportError:
            raise ImportError("matplotlib is required for plot()")

        fig, ax = plt.subplots(figsize=figsize)

        order = self._compute_topological_order()

        # Assign positions: arrange by topological depth
        depths = {}
        for name in order:
            parents = self.get_parents(name)
            if not parents:
                depths[name] = 0
            else:
                depths[name] = max(depths[p] for p in parents) + 1

        # Group nodes by depth
        depth_groups = {}
        for name, d in depths.items():
            depth_groups.setdefault(d, []).append(name)

        max_depth = max(depths.values()) if depths else 0

        # Compute positions
        positions = {}
        for d, nodes in depth_groups.items():
            n = len(nodes)
            for i, name in enumerate(nodes):
                x = (i + 0.5) / n if n > 0 else 0.5
                y = 1 - d / (max_depth + 1) if max_depth > 0 else 0.5
                positions[name] = (x, y)

        # Draw edges
        for target, edges in self.edges.items():
            for edge in edges:
                x1, y1 = positions[edge.source]
                x2, y2 = positions[target]

                # Draw arrow
                ax.annotate(
                    "",
                    xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="gray",
                        lw=1.5,
                        connectionstyle="arc3,rad=0.1"
                    )
                )

                # Add weight label
                if show_weights:
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    label = f"{edge.weight:.2f}"
                    if not isinstance(edge.transform, IdentityTransform):
                        label += f"\n({edge.transform.__class__.__name__[:4]})"
                    ax.text(mid_x + 0.02, mid_y, label, fontsize=8, color="darkblue")

        # Draw nodes
        for name, (x, y) in positions.items():
            node = self.nodes[name]
            color = "lightcoral" if node.intervened else "lightblue"
            circle = Circle((x, y), 0.05, color=color, ec="black", lw=2, zorder=10)
            ax.add_patch(circle)
            ax.text(x, y, name, ha="center", va="center", fontsize=10, fontweight="bold", zorder=11)

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("DAG Structure")

        plt.tight_layout()
        return fig, ax

    def to_ascii(self) -> str:
        """
        Return a simple ASCII representation of the DAG.

        Returns:
            ASCII string showing the graph structure
        """
        lines = []
        order = self._compute_topological_order()

        for name in order:
            parents = self.get_parents(name)
            if parents:
                for parent in parents:
                    edge = next(e for e in self.edges[name] if e.source == parent)
                    weight_str = f"({edge.weight:.2f})"
                    transform_str = ""
                    if not isinstance(edge.transform, IdentityTransform):
                        transform_str = f" [{edge.transform.__class__.__name__}]"
                    lines.append(f"  {parent} --{weight_str}{transform_str}--> {name}")

        if not lines:
            lines = [f"  {name} (isolated)" for name in order]

        return "DAG:\n" + "\n".join(lines)
