"""Classification data generator using DAG-based structural models."""

from __future__ import annotations

import numpy as np
from typing import Optional, Union, Literal
from dataclasses import dataclass, field
import warnings

from .noise import NoiseGenerator, GaussianNoise
from .transforms import Transform, IdentityTransform, get_transform
from .dag import DAG


@dataclass
class FeatureSpec:
    """Specification for a single feature in classification data generation."""
    name: str
    # For generative mode: location shift based on class (loc_class_0, loc_class_1)
    loc_by_class: tuple[float, float] = (0.0, 0.0)
    # For causal mode: weight contribution to Y
    weight_to_y: float = 0.0
    transform_to_y: Optional[Union[str, Transform]] = None
    # Noise settings
    noise_std: float = 1.0
    noise: Optional[NoiseGenerator] = None
    # Dependencies on other features
    parents: list[str] = field(default_factory=list)
    parent_weights: list[float] = field(default_factory=list)
    parent_transforms: list[Union[str, Transform]] = field(default_factory=list)
    # Final transform applied to the feature value
    output_transform: Optional[Union[str, Transform]] = None

    def __post_init__(self):
        """Ensure default weights and transforms for parents are set."""
        if self.parents and not self.parent_weights:
            self.parent_weights = [1.0] * len(self.parents)
        if self.parents and not self.parent_transforms:
            self.parent_transforms = ["linear"] * len(self.parents)


class ClassificationDataGenerator:
    """
    Generate synthetic classification data using a DAG-based approach.

    Supports two modes:

    1. **Generative mode** (`mode="generative"`):
       - Y is sampled first from Bernoulli(class_balance)
       - Features X are generated conditional on Y: X_i = loc[y] + f(parents) + noise
       - Gives direct control over class balance
       - Useful for creating well-separated classes

    2. **Causal mode** (`mode="causal"`):
       - Features X are generated from the DAG structure
       - Y is computed as a function of X: P(Y=1|X) = sigmoid(sum of weighted features)
       - Reflects true causal structure (X causes Y)
       - Class balance controlled via intercept parameter

    Example (generative mode):
        >>> gen = ClassificationDataGenerator(
        ...     mode="generative",
        ...     class_balance=0.1,
        ...     feature_specs=[
        ...         FeatureSpec("f0", loc_by_class=(0.0, 2.0), noise_std=1.0),
        ...         FeatureSpec("f1", loc_by_class=(-0.5, 1.0), noise_std=1.0),
        ...         FeatureSpec("f2", parents=["f0", "f1"], parent_weights=[1.0, 0.5],
        ...                     output_transform="tanh", noise_std=0.5),
        ...     ],
        ...     n_noise_features=3
        ... )
        >>> X, y = gen.generate_batch(1000)

    Example (causal mode):
        >>> gen = ClassificationDataGenerator(
        ...     mode="causal",
        ...     class_balance=0.1,  # Target class balance (adjusts intercept)
        ...     feature_specs=[
        ...         FeatureSpec("f0", noise_std=1.0, weight_to_y=1.5),
        ...         FeatureSpec("f1", noise_std=1.0, weight_to_y=0.8),
        ...         FeatureSpec("f2", parents=["f0", "f1"], parent_weights=[1.0, 0.5],
        ...                     output_transform="tanh", noise_std=0.5, weight_to_y=0.5),
        ...     ],
        ...     n_noise_features=3
        ... )
        >>> X, y = gen.generate_batch(1000)
    """

    def __init__(
        self,
        mode: Literal["generative", "causal"] = "generative",
        class_balance: float = 0.5,
        feature_specs: Optional[list[FeatureSpec]] = None,
        n_noise_features: int = 0,
        noise_feature_std: float = 1.0,
        # Causal mode specific parameters
        link_function: Literal["logistic", "probit", "linear"] = "logistic",
        intercept: Optional[float] = None,
        label_noise: float = 0.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize the classification data generator.

        Args:
            mode: "generative" (Y first, then X|Y) or "causal" (X first, then Y|X)
            class_balance: P(y=1), the target proportion of positive class
            feature_specs: List of FeatureSpec defining each informative feature
            n_noise_features: Number of pure noise features to add
            noise_feature_std: Std dev for noise features
            link_function: For causal mode - "logistic", "probit", or "linear"
            intercept: For causal mode - intercept term. If None, auto-calibrated to match class_balance
            label_noise: Probability of flipping the label (for both modes)
            seed: Random seed for reproducibility
        """
        self.mode = mode
        self.class_balance = class_balance
        self.feature_specs = feature_specs or []
        self.n_noise_features = n_noise_features
        self.noise_feature_std = noise_feature_std
        self.link_function = link_function
        self.intercept = intercept
        self.label_noise = label_noise
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self._build_dag()

        # For causal mode: auto-calibrate intercept if not provided
        if self.mode == "causal" and self.intercept is None:
            self._calibrate_intercept()

    def _build_dag(self):
        """Build internal DAG to determine feature generation order."""
        self.dag = DAG()

        for spec in self.feature_specs:
            noise = spec.noise or GaussianNoise(std=spec.noise_std)
            self.dag.add_node(spec.name, noise=noise)

        for spec in self.feature_specs:
            for i, parent in enumerate(spec.parents):
                if parent not in self.dag.nodes:
                    raise ValueError(f"Feature '{spec.name}' depends on unknown feature '{parent}'")
                weight = spec.parent_weights[i] if i < len(spec.parent_weights) else 1.0
                transform = spec.parent_transforms[i] if i < len(spec.parent_transforms) else "linear"
                self.dag.add_edge(parent, spec.name, weight=weight, transform=transform)

        self._feature_order = self.dag._compute_topological_order()
        self._spec_lookup = {spec.name: spec for spec in self.feature_specs}

    def _calibrate_intercept(self, n_samples: int = 10000):
        """
            Auto-calibrate intercept to achieve target class balance in causal mode.

            An interecept is added to the structural equations for Y.
                η = intercept + Σ(weight_i × transform_i(X_i)) 
            where η is the linear predictor.
            We find intercept such that mean(link_function(η)) ≈ class_balance by sampling. 
        """
        # Generate features without labels to estimate required intercept
        temp_rng = np.random.default_rng(42)  # Fixed seed for calibration
        features = self._generate_features(n_samples, temp_rng)

        # Compute linear predictor (without intercept)
        linear_pred = self._compute_linear_predictor(features, intercept=0.0)

        # Find intercept that gives target class balance mean(P(Y=1)) = class_balance
        # Using bisection search
        from scipy.optimize import brentq

        def balance_error(intercept):
            probs = self._apply_link(linear_pred + intercept)
            return probs.mean() - self.class_balance

        try:
            self.intercept = brentq(balance_error, -20, 20)
        except ValueError:
            # If brentq fails, use a reasonable default
            warnings.warn("Could not calibrate intercept precisely. Using approximate value.")
            self.intercept = np.log(self.class_balance / (1 - self.class_balance + 1e-10))

    def _generate_features(
        self,
        n: int,
        rng: np.random.Generator,
        y: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """Generate feature values."""
        features = {}

        for name in self._feature_order:
            spec = self._spec_lookup[name]

            if self.mode == "generative" and y is not None:
                # Generative mode: class-conditional location
                loc = np.where(y == 1, spec.loc_by_class[1], spec.loc_by_class[0])
                value = loc.copy()
            else:
                # Causal mode: start from zero
                value = np.zeros(n)

            # Add parent contributions
            for edge in self.dag.edges.get(name, []):
                parent_value = features[edge.source]
                value = value + edge.compute_contribution(parent_value)

            # Add noise
            noise_gen = spec.noise or GaussianNoise(std=spec.noise_std)
            value = value + noise_gen.sample(n, rng)

            # Apply output transform if specified
            if spec.output_transform is not None:
                if isinstance(spec.output_transform, str):
                    transform = get_transform(spec.output_transform)
                else:
                    transform = spec.output_transform
                value = transform(value)

            features[name] = value

        return features

    def _compute_linear_predictor(
        self,
        features: dict[str, np.ndarray],
        intercept: Optional[float] = None,
    ) -> np.ndarray:
        """
            Compute linear predictor for causal mode: η = intercept + Σ(weight_i × transform_i(X_i)).
            Use to compute probabilities for Y using a link function.
        """
        if intercept is None:
            intercept = self.intercept or 0.0

        linear_pred = np.full(len(next(iter(features.values()))), intercept, dtype=float)

        for spec in self.feature_specs:
            if spec.weight_to_y != 0:
                feat_value = features[spec.name]
                if spec.transform_to_y is not None:
                    if isinstance(spec.transform_to_y, str):
                        transform = get_transform(spec.transform_to_y)
                    else:
                        transform = spec.transform_to_y
                    feat_value = transform(feat_value)
                linear_pred = linear_pred + spec.weight_to_y * feat_value

        return linear_pred

    def _apply_link(self, linear_pred: np.ndarray) -> np.ndarray:
        """
            Apply link function to get probabilities from real numbers.
        """
        if self.link_function == "logistic":
            return 1.0 / (1.0 + np.exp(-np.clip(linear_pred, -500, 500)))
        elif self.link_function == "probit":
            from scipy.stats import norm
            return norm.cdf(linear_pred)
        elif self.link_function == "linear":
            return np.clip(linear_pred, 0, 1)
        else:
            raise ValueError(f"Unknown link function: {self.link_function}")

    def generate_batch(
        self,
        n: int,
        return_dataframe: bool = False,
        random_state: Optional[int] = None,
    ) -> Union[tuple[np.ndarray, np.ndarray], "pd.DataFrame"]:
        """
        Generate a batch of classification data.

        Args:
            n: Number of samples to generate
            return_dataframe: If True, return a pandas DataFrame instead of arrays
            random_state: Override the random state for this batch

        Returns:
            If return_dataframe=False: tuple (X, y) where X is (n, n_features) and y is (n,)
            If return_dataframe=True: DataFrame with features and 'y' column
        """
        if random_state is not None:
            rng = np.random.default_rng(random_state)
        else:
            rng = self.rng

        if self.mode == "generative":
            # Generative mode: Y first, then X|Y
            y = rng.binomial(1, self.class_balance, size=n)
            features = self._generate_features(n, rng, y=y)
        else:
            # Causal mode: X first, then Y|X
            features = self._generate_features(n, rng, y=None)
            linear_pred = self._compute_linear_predictor(features)
            probs = self._apply_link(linear_pred)
            y = rng.binomial(1, probs)

        # Apply label noise if specified
        if self.label_noise > 0:
            flip_mask = rng.random(n) < self.label_noise
            y = np.where(flip_mask, 1 - y, y)

        # Add noise features
        noise_features = {}
        for i in range(self.n_noise_features):
            noise_features[f"noise_{i}"] = rng.normal(0, self.noise_feature_std, n)

        # Combine into feature matrix
        feature_names = self._feature_order + list(noise_features.keys())
        X = np.column_stack(
            [features[name] for name in self._feature_order] +
            [noise_features[name] for name in noise_features]
        )

        if return_dataframe:
            try:
                import pandas as pd
            except ImportError:
                raise ImportError("pandas is required for return_dataframe=True")
            df = pd.DataFrame(X, columns=feature_names)
            df['y'] = y
            return df

        return X, y

    def get_feature_names(self) -> list[str]:
        """Get the names of all features in order."""
        return self._feature_order + [f"noise_{i}" for i in range(self.n_noise_features)]

    def get_full_dag(self) -> DAG:
        """
        Get the full DAG including Y as a node.

        Returns a DAG that includes:
        - All feature nodes
        - Y as a distinct target node
        - Edges from features to Y (for causal mode) or from Y to features (for generative mode)
        """
        full_dag = self.dag.copy()

        # Add Y node
        full_dag.add_node("Y", noise_std=0.0)

        if self.mode == "causal":
            # Causal: features -> Y
            for spec in self.feature_specs:
                if spec.weight_to_y != 0:
                    transform = spec.transform_to_y if spec.transform_to_y else "linear"
                    full_dag.add_edge(spec.name, "Y", weight=spec.weight_to_y, transform=transform)
        else:
            # Generative: Y -> features (conceptually Y influences feature distributions)
            for spec in self.feature_specs:
                if spec.loc_by_class[0] != spec.loc_by_class[1]:
                    # There's a class-conditional shift
                    shift = spec.loc_by_class[1] - spec.loc_by_class[0]
                    full_dag.add_edge("Y", spec.name, weight=shift)

        return full_dag

    def plot_dag(
        self,
        figsize: tuple[float, float] = (10, 8),
        show_weights: bool = True,
        target_name: str = "Y",
    ):
        """
        Plot the full DAG including the target variable Y.

        Y is displayed as a distinct node (colored differently) to make
        the classification structure clear.

        Args:
            figsize: Figure size (width, height)
            show_weights: Whether to display edge weights
            target_name: Label for the target node (default "Y")

        Returns:
            matplotlib figure and axes
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
            from matplotlib.lines import Line2D
        except ImportError:
            raise ImportError("matplotlib is required for plot_dag()")

        fig, ax = plt.subplots(figsize=figsize)

        # Get feature order and build positions
        feature_order = self._feature_order

        # Compute depths for features
        depths = {}
        for name in feature_order:
            parents = self.dag.get_parents(name)
            if not parents:
                depths[name] = 0
            else:
                depths[name] = max(depths[p] for p in parents) + 1

        # Group features by depth
        depth_groups = {}
        for name, d in depths.items():
            depth_groups.setdefault(d, []).append(name)

        max_depth = max(depths.values()) if depths else 0

        # Position features
        positions = {}
        for d, nodes in depth_groups.items():
            n = len(nodes)
            for i, name in enumerate(nodes):
                x = (i + 0.5) / max(n, 1)
                if self.mode == "causal":
                    # Features at top, Y at bottom
                    y = 1 - d / (max_depth + 2)
                else:
                    # Y at top, features below
                    y = 1 - (d + 1) / (max_depth + 2)
                positions[name] = (x, y)

        # Position Y
        if self.mode == "causal":
            positions[target_name] = (0.5, 0.05)  # Y at bottom center
        else:
            positions[target_name] = (0.5, 0.95)  # Y at top center

        # Draw feature-to-feature edges
        for target, edges in self.dag.edges.items():
            for edge in edges:
                x1, y1 = positions[edge.source]
                x2, y2 = positions[target]
                ax.annotate(
                    "",
                    xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5,
                                   connectionstyle="arc3,rad=0.1")
                )
                if show_weights:
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    label = f"{edge.weight:.2f}"
                    if not isinstance(edge.transform, IdentityTransform):
                        label += f"\n({edge.transform.__class__.__name__[:4]})"
                    ax.text(mid_x + 0.02, mid_y, label, fontsize=7, color="darkblue")

        # Draw edges to/from Y
        if self.mode == "causal":
            # Features -> Y
            for spec in self.feature_specs:
                if spec.weight_to_y != 0:
                    x1, y1 = positions[spec.name]
                    x2, y2 = positions[target_name]
                    ax.annotate(
                        "",
                        xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", color="darkgreen", lw=2,
                                       connectionstyle="arc3,rad=0.05")
                    )
                    if show_weights:
                        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                        label = f"{spec.weight_to_y:.2f}"
                        if spec.transform_to_y:
                            t_name = spec.transform_to_y if isinstance(spec.transform_to_y, str) else spec.transform_to_y.__class__.__name__[:4]
                            label += f"\n({t_name[:4]})"
                        ax.text(mid_x + 0.03, mid_y, label, fontsize=7, color="darkgreen")
        else:
            # Y -> Features (generative)
            for spec in self.feature_specs:
                if spec.loc_by_class[0] != spec.loc_by_class[1]:
                    x1, y1 = positions[target_name]
                    x2, y2 = positions[spec.name]
                    ax.annotate(
                        "",
                        xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", color="darkgreen", lw=2,
                                       connectionstyle="arc3,rad=0.05")
                    )
                    if show_weights:
                        shift = spec.loc_by_class[1] - spec.loc_by_class[0]
                        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                        ax.text(mid_x + 0.03, mid_y, f"{shift:.2f}", fontsize=7, color="darkgreen")

        # Draw feature nodes
        for name in feature_order:
            x, y = positions[name]
            circle = Circle((x, y), 0.04, color="lightblue", ec="black", lw=2, zorder=10)
            ax.add_patch(circle)
            ax.text(x, y, name, ha="center", va="center", fontsize=9, fontweight="bold", zorder=11)

        # Draw Y node (larger, different color)
        x, y = positions[target_name]
        y_circle = Circle((x, y), 0.06, color="gold", ec="darkgoldenrod", lw=3, zorder=10)
        ax.add_patch(y_circle)
        ax.text(x, y, target_name, ha="center", va="center", fontsize=12, fontweight="bold", zorder=11)

        # Add legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
                   markersize=10, label='Features'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gold',
                   markersize=12, label='Target (Y)'),
            Line2D([0], [0], color='gray', lw=1.5, label='Feature edges'),
            Line2D([0], [0], color='darkgreen', lw=2, label='Target edges'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect("equal")
        ax.axis("off")

        mode_str = "Causal (X → Y)" if self.mode == "causal" else "Generative (Y → X)"
        ax.set_title(f"Classification DAG - {mode_str}", fontsize=12)

        plt.tight_layout()
        return fig, ax

    def describe(self) -> str:
        """Return a description of the data generation process."""
        lines = [
            "ClassificationDataGenerator",
            "=" * 40,
            f"Mode: {self.mode}",
            f"Class balance P(y=1): {self.class_balance}",
            f"Number of informative features: {len(self.feature_specs)}",
            f"Number of noise features: {self.n_noise_features}",
        ]

        if self.mode == "causal":
            lines.append(f"Link function: {self.link_function}")
            lines.append(f"Intercept: {self.intercept}")

        if self.label_noise > 0:
            lines.append(f"Label noise: {self.label_noise}")

        lines.append("")
        lines.append("Feature specifications:")

        for spec in self.feature_specs:
            lines.append(f"\n  {spec.name}:")
            if self.mode == "generative":
                lines.append(f"    loc_by_class: {spec.loc_by_class}")
            if self.mode == "causal" and spec.weight_to_y != 0:
                lines.append(f"    weight_to_y: {spec.weight_to_y}")
                if spec.transform_to_y:
                    lines.append(f"    transform_to_y: {spec.transform_to_y}")
            lines.append(f"    noise_std: {spec.noise_std}")
            if spec.parents:
                lines.append(f"    parents: {spec.parents}")
                lines.append(f"    parent_weights: {spec.parent_weights}")
                lines.append(f"    parent_transforms: {spec.parent_transforms}")
            if spec.output_transform:
                lines.append(f"    output_transform: {spec.output_transform}")

        return "\n".join(lines)

    @classmethod
    def from_random(
        cls,
        n_features: int = 6,
        n_informative: int = 4,
        n_direct_to_y: int = 2,
        connectivity: float = 0.3,
        class_balance: float = 0.5,
        mode: Literal["generative", "causal"] = "generative",
        # Weight and transform settings
        weight_range: tuple[float, float] = (0.5, 1.5),
        weight_to_y_range: tuple[float, float] = (0.5, 2.0),
        loc_shift_range: tuple[float, float] = (0.5, 2.0),
        noise_std_range: tuple[float, float] = (0.5, 1.5),
        nonlinear_prob: float = 0.2,
        available_transforms: Optional[list[str]] = None,
        # Other settings
        link_function: Literal["logistic", "probit", "linear"] = "logistic",
        label_noise: float = 0.0,
        seed: Optional[int] = None,
    ) -> "ClassificationDataGenerator":
        """
        Create a ClassificationDataGenerator with randomly generated feature structure.

        This factory method generates a DAG with specified high-level properties,
        making it easy to create complex classification scenarios without manually
        specifying each feature.

        Args:
            n_features: Total number of features (informative + noise)
            n_informative: Number of informative features (with signal)
            n_direct_to_y: Number of features directly connected to Y (causal mode)
                          or with class-conditional shift (generative mode)
            connectivity: Probability of edge between any two informative features
                         (controls feature dependencies, 0=independent, 1=fully connected)
            class_balance: P(y=1), target proportion of positive class
            mode: "generative" or "causal"
            weight_range: Range for edge weights between features
            weight_to_y_range: Range for feature weights to Y (causal mode)
            loc_shift_range: Range for location shifts (generative mode)
            noise_std_range: Range for noise standard deviations
            nonlinear_prob: Probability that an edge has a non-linear transform
            available_transforms: List of transform names to sample from
            link_function: Link function for causal mode
            label_noise: Probability of flipping labels
            seed: Random seed for reproducibility

        Returns:
            Configured ClassificationDataGenerator

        Example:
            >>> gen = ClassificationDataGenerator.from_random(
            ...     n_features=10,
            ...     n_informative=6,
            ...     n_direct_to_y=3,
            ...     connectivity=0.4,
            ...     class_balance=0.2,
            ...     mode="causal",
            ...     seed=42
            ... )
            >>> X, y = gen.generate_batch(1000)
            >>> print(gen.describe())
        """
        rng = np.random.default_rng(seed)

        if n_informative > n_features:
            raise ValueError("n_informative cannot exceed n_features")
        if n_direct_to_y > n_informative:
            raise ValueError("n_direct_to_y cannot exceed n_informative")

        available_transforms = available_transforms or ["quadratic", "tanh", "sigmoid", "relu"]
        n_noise = n_features - n_informative

        # Generate feature names
        feature_names = [f"f{i}" for i in range(n_informative)]

        # Decide which features connect directly to Y
        direct_to_y_indices = set(rng.choice(n_informative, size=n_direct_to_y, replace=False))

        # Build adjacency for feature dependencies (only earlier -> later to ensure DAG)
        feature_specs = []

        for i, name in enumerate(feature_names):
            # Determine parents (only from earlier features)
            parents = []
            parent_weights = []
            parent_transforms = []

            for j in range(i):
                if rng.random() < connectivity:
                    parents.append(feature_names[j])
                    weight = rng.uniform(weight_range[0], weight_range[1])
                    if rng.random() < 0.5:
                        weight = -weight  # Allow negative weights
                    parent_weights.append(weight)

                    if rng.random() < nonlinear_prob:
                        transform = rng.choice(available_transforms)
                    else:
                        transform = "linear"
                    parent_transforms.append(transform)

            # Noise std
            noise_std = rng.uniform(noise_std_range[0], noise_std_range[1])

            # Output transform (occasional)
            output_transform = None
            if rng.random() < nonlinear_prob * 0.5:
                output_transform = rng.choice(available_transforms)

            # Mode-specific settings
            if mode == "generative":
                # Location shift for direct-to-y features
                if i in direct_to_y_indices:
                    shift = rng.uniform(loc_shift_range[0], loc_shift_range[1])
                    if rng.random() < 0.5:
                        # Positive class has higher values
                        loc_by_class = (0.0, shift)
                    else:
                        # Positive class has lower values
                        loc_by_class = (shift, 0.0)
                else:
                    loc_by_class = (0.0, 0.0)
                weight_to_y = 0.0
                transform_to_y = None
            else:
                # Causal mode: weight to Y for direct features
                loc_by_class = (0.0, 0.0)
                if i in direct_to_y_indices:
                    weight_to_y = rng.uniform(weight_to_y_range[0], weight_to_y_range[1])
                    if rng.random() < 0.5:
                        weight_to_y = -weight_to_y
                    if rng.random() < nonlinear_prob:
                        transform_to_y = rng.choice(available_transforms)
                    else:
                        transform_to_y = None
                else:
                    weight_to_y = 0.0
                    transform_to_y = None

            spec = FeatureSpec(
                name=name,
                loc_by_class=loc_by_class,
                weight_to_y=weight_to_y,
                transform_to_y=transform_to_y,
                noise_std=noise_std,
                parents=parents,
                parent_weights=parent_weights,
                parent_transforms=parent_transforms,
                output_transform=output_transform,
            )
            feature_specs.append(spec)

        return cls(
            mode=mode,
            class_balance=class_balance,
            feature_specs=feature_specs,
            n_noise_features=n_noise,
            noise_feature_std=rng.uniform(noise_std_range[0], noise_std_range[1]),
            link_function=link_function,
            label_noise=label_noise,
            seed=seed,
        )
