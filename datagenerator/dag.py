"""Directed Acyclic Graph for structural causal models."""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .categorical import CategoricalNodeInfo
from .noise import GaussianNoise, LaplacianNoise, NoiseGenerator, StudentTNoise, UniformNoise
from .transforms import IdentityTransform, Transform, get_transform


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
    intervention_value: float | None = None


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
        """Initialize DAG with nodes and edges dict."""
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, list[Edge]] = defaultdict(list)  # target -> list of incoming edges
        self._topological_order: list[str] | None = None
        self.categorical_nodes: dict[str, CategoricalNodeInfo] = {}
        self._sub_node_to_categorical: dict[str, str] = {}  # reverse lookup cache

    def add_node(
        self,
        name: str,
        noise: NoiseGenerator | None = None,
        noise_type: str = "gaussian",
        noise_std: float = 1.0,
        noise_params: dict | None = None,
    ) -> DAG:
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
            warnings.warn(f"Node '{name}' already exists, updating it.", stacklevel=2)

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

    def add_categorical_node(
        self,
        name: str,
        categories: list[str],
        probabilities: list[float] | None = None,
    ) -> DAG:
        """Add a categorical variable, internally expanded into one-hot sub-nodes.

        Args:
            name: User-facing name for the categorical variable.
            categories: List of category labels (e.g., ["R", "G", "B"]).
            probabilities: Sampling probabilities for root categorical nodes.
                If None, the node is a child whose distribution comes from parents via softmax.

        Returns:
            Self for method chaining.
        """
        if name in self.nodes or name in self.categorical_nodes:
            raise ValueError(f"Node '{name}' already exists in the DAG.")

        info = CategoricalNodeInfo(name=name, categories=categories, probabilities=probabilities)
        self.categorical_nodes[name] = info

        # Add one-hot sub-nodes with zero-std noise (placeholder, never used during sampling)
        for sub_name in info.sub_node_names:
            self.add_node(sub_name, noise_std=0.0)
            self._sub_node_to_categorical[sub_name] = name

        return self

    def is_categorical(self, name: str) -> bool:
        """Check if a node name corresponds to a categorical variable."""
        return name in self.categorical_nodes

    def get_categorical_owner(self, sub_node_name: str) -> str | None:
        """Return the categorical variable name that owns this sub-node, or None."""
        return self._sub_node_to_categorical.get(sub_node_name)

    def add_edge(
        self,
        source: str,
        target: str,
        weight: float = 1.0,
        transform: Transform | str | None = None,
        transform_params: dict | None = None,
        weights: dict[str, float] | None = None,
        transforms: dict[str, str | Transform] | None = None,
    ) -> DAG:
        """Add a directed edge from source to target.

        For edges involving categorical variables, use the ``weights`` and
        ``transforms`` dicts to specify per-category weights and transforms.

        Args:
            source: Name of the parent node (may be categorical).
            target: Name of the child node (may be categorical).
            weight: Scalar weight (used only for continuous→continuous edges).
            transform: Transform instance or name (continuous→continuous edges).
            transform_params: Parameters for named transforms.
            weights: Per-category weights, e.g. {"R": 2.0, "G": -1.0}.
                Required when source or target is categorical.
            transforms: Per-category transforms, e.g. {"R": "quadratic"}.
                Missing keys default to identity.

        Returns:
            Self for method chaining.
        """
        source_is_cat = self.is_categorical(source)
        target_is_cat = self.is_categorical(target)

        if source_is_cat and target_is_cat:
            raise NotImplementedError("Categorical-to-categorical edges are not yet supported.")

        # Dispatch to categorical edge logic if either endpoint is categorical
        if source_is_cat or target_is_cat:
            return self._add_categorical_edge(
                source, target, source_is_cat, target_is_cat, weights, transforms, transform_params
            )

        # --- Standard continuous→continuous edge ---
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

    def _add_categorical_edge(
        self,
        source: str,
        target: str,
        source_is_cat: bool,
        target_is_cat: bool,
        weights: dict[str, float] | None,
        transforms: dict[str, str | Transform] | None,
        transform_params: dict | None,
    ) -> DAG:
        """Internal: create per-category sub-edges for categorical endpoints."""
        transforms = transforms or {}
        transform_params = transform_params or {}

        if source_is_cat:
            # Categorical parent → continuous child: one edge per source sub-node
            info = self.categorical_nodes[source]
            if target not in self.nodes:
                self.add_node(target)
            weights = weights or {cat: 0.0 for cat in info.categories}

            for cat in info.categories:
                sub_name = f"{source}_{cat}"
                w = weights.get(cat, 0.0)
                t = transforms.get(cat, "identity")
                if isinstance(t, str):
                    t = get_transform(t, **transform_params)
                self.nodes[target].is_root = False
                edge = Edge(source=sub_name, target=target, weight=w, transform=t)
                self.edges[target].append(edge)

        else:
            # Continuous parent → categorical child: one edge per target sub-node
            info = self.categorical_nodes[target]
            if source not in self.nodes:
                self.add_node(source)
            weights = weights or {cat: 0.0 for cat in info.categories}

            for cat in info.categories:
                sub_name = f"{target}_{cat}"
                w = weights.get(cat, 0.0)
                t = transforms.get(cat, "identity")
                if isinstance(t, str):
                    t = get_transform(t, **transform_params)
                self.nodes[sub_name].is_root = False
                edge = Edge(source=source, target=sub_name, weight=w, transform=t)
                self.edges[sub_name].append(edge)

        # Check for cycles
        self._topological_order = None
        try:
            self._compute_topological_order()
        except ValueError as e:
            raise ValueError(f"Adding edge {source} -> {target} would create a cycle") from e

        return self

    def _compute_topological_order(self) -> list[str]:
        """Compute topological ordering using Kahn's algorithm.

        It:
            - computes the order in which nodes must be processed (parents before children).
              Needed for sampling.
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

    def intervene(self, node: str, value: float | str) -> DAG:
        """Set an intervention on a node (do-calculus).

        This fixes the node to a constant value, breaking the influence
        of its parents.

        For categorical nodes, pass a category name as ``value``
        (e.g., ``dag.intervene("Color", "R")``).

        Args:
            node: Name of the node to intervene on.
            value: Fixed value (float for continuous, category name for categorical).

        Returns:
            Self for method chaining.
        """
        if self.is_categorical(node):
            info = self.categorical_nodes[node]
            if value not in info.categories:
                raise ValueError(f"'{value}' is not a valid category for '{node}'. Options: {info.categories}")
            for sub_name in info.sub_node_names:
                cat = sub_name.removeprefix(f"{node}_")
                self.nodes[sub_name].intervened = True
                self.nodes[sub_name].intervention_value = 1.0 if cat == value else 0.0
            return self

        if node not in self.nodes:
            raise ValueError(f"Node '{node}' not in DAG")
        if isinstance(value, str):
            raise TypeError(f"String value '{value}' is only valid for categorical nodes, but '{node}' is continuous.")
        self.nodes[node].intervened = True
        self.nodes[node].intervention_value = value
        return self

    def clear_interventions(self) -> DAG:
        """Clear all interventions."""
        for node in self.nodes.values():
            node.intervened = False
            node.intervention_value = None
        return self

    def copy(self) -> DAG:
        """Create a deep copy of this DAG."""
        import copy

        return copy.deepcopy(self)

    def __repr__(self) -> str:
        """Return a concise string representation of the DAG."""
        n_nodes = len(self.nodes)
        n_edges = sum(len(e) for e in self.edges.values())
        return f"DAG(nodes={n_nodes}, edges={n_edges})"

    def _get_user_facing_order(self) -> list[str]:
        """Return topological order with sub-nodes collapsed to categorical names."""
        order = self._compute_topological_order()
        result: list[str] = []
        seen_categoricals: set[str] = set()
        for name in order:
            cat_name = self.get_categorical_owner(name)
            if cat_name is not None:
                if cat_name not in seen_categoricals:
                    seen_categoricals.add(cat_name)
                    result.append(cat_name)
            else:
                result.append(name)
        return result

    def describe(self) -> str:
        """Return a detailed description of the DAG structure."""
        lines = ["DAG Structure:", "=" * 40]

        for name in self._get_user_facing_order():
            if self.is_categorical(name):
                info = self.categorical_nodes[name]
                # Check if it's a root categorical
                has_parents = any(len(self.edges.get(sn, [])) > 0 for sn in info.sub_node_names)
                if has_parents:
                    # Collect unique parent names (resolving sub-nodes to categorical names)
                    parent_names: set[str] = set()
                    for sn in info.sub_node_names:
                        for edge in self.edges.get(sn, []):
                            owner = self.get_categorical_owner(edge.source)
                            parent_names.add(owner if owner else edge.source)
                    lines.append(f"\n{name} [categorical: {info.categories}] <- {', '.join(sorted(parent_names))}")
                    for sn in info.sub_node_names:
                        for edge in self.edges.get(sn, []):
                            lines.append(f"  {edge.source} -> {sn}: weight={edge.weight}, transform={edge.transform}")
                else:
                    lines.append(f"\n{name} [categorical root: {info.categories}]")
                    lines.append(f"  probabilities: {info.probabilities}")
                # Show intervention status
                if any(self.nodes[sn].intervened for sn in info.sub_node_names):
                    active = [sn for sn in info.sub_node_names if self.nodes[sn].intervention_value == 1.0]
                    cat_val = active[0].removeprefix(f"{name}_") if active else "?"
                    lines.append(f"  INTERVENED: {cat_val}")
            else:
                node = self.nodes[name]
                parents = self.get_parents(name)
                if parents:
                    # Resolve sub-node parents to categorical names for display
                    display_parents = []
                    for p in parents:
                        owner = self.get_categorical_owner(p)
                        display_parents.append(owner if owner else p)
                    parent_str = ", ".join(dict.fromkeys(display_parents))
                    lines.append(f"\n{name} <- {parent_str}")
                    for edge in self.edges[name]:
                        lines.append(f"  {edge.source}: weight={edge.weight}, transform={edge.transform}")
                else:
                    lines.append(f"\n{name} (root)")
                lines.append(f"  noise: {node.noise}")
                if node.intervened:
                    lines.append(f"  INTERVENED: {node.intervention_value}")

        return "\n".join(lines)

    def show_equations(self) -> str:
        """Return the structural equations in mathematical notation."""
        lines = ["Structural Equations:", "=" * 40]

        for name in self._get_user_facing_order():
            if self.is_categorical(name):
                info = self.categorical_nodes[name]
                has_parents = any(len(self.edges.get(sn, [])) > 0 for sn in info.sub_node_names)

                # Check intervention
                if any(self.nodes[sn].intervened for sn in info.sub_node_names):
                    active = [
                        cat
                        for sn, cat in zip(info.sub_node_names, info.categories, strict=True)
                        if self.nodes[sn].intervention_value == 1.0
                    ]
                    lines.append(f"{name} = {active[0] if active else '?'}  [intervened]")
                elif not has_parents:
                    # Root categorical
                    prob_parts = [
                        f"{cat}={p:.2f}" for cat, p in zip(info.categories, info.probabilities or [], strict=True)
                    ]
                    lines.append(f"{name} ~ Multinomial({', '.join(prob_parts)})")
                else:
                    # Child categorical: show score equations + softmax
                    score_lines = []
                    for cat, sub_name in zip(info.categories, info.sub_node_names, strict=True):
                        edges = self.edges.get(sub_name, [])
                        terms = []
                        for edge in edges:
                            transformed = edge.transform.math_notation(edge.source)
                            if edge.weight == 1.0:
                                terms.append(transformed)
                            elif edge.weight == -1.0:
                                terms.append(f"-{transformed}")
                            else:
                                terms.append(f"{edge.weight:.2f} * {transformed}")
                        score_lines.append(f"  score_{cat} = {' + '.join(terms)}")
                    lines.append(f"{name} ~ Softmax(scores)  where:")
                    lines.extend(score_lines)
            else:
                node = self.nodes[name]

                if node.intervened:
                    lines.append(f"{name} = {node.intervention_value}  [intervened]")
                    continue

                incoming_edges = self.edges.get(name, [])

                if not incoming_edges:
                    lines.append(f"{name} = {node.noise.math_notation()}")
                else:
                    terms = []
                    for edge in incoming_edges:
                        transformed = edge.transform.math_notation(edge.source)
                        if edge.weight == 1.0:
                            terms.append(transformed)
                        elif edge.weight == -1.0:
                            terms.append(f"-{transformed}")
                        else:
                            terms.append(f"{edge.weight:.2f} * {transformed}")
                    equation = " + ".join(terms)
                    lines.append(f"{name} = {equation} + {node.noise.math_notation()}")

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
            raise ImportError("matplotlib is required for plot()") from None

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
                    xy=(x2, y2),
                    xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, connectionstyle="arc3,rad=0.1"),
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
