# DataGenerator Repository - Complete Guide

This is a Python package for generating **synthetic data** based on **Structural Causal Models (SCMs)**. It's designed for researchers and practitioners who need controlled synthetic datasets for testing machine learning algorithms, causal inference methods, or understanding data generation processes.

---

## 1. Core Concepts

### What is a Structural Causal Model (SCM)?

An SCM represents causal relationships between variables using:
1. **A DAG (Directed Acyclic Graph)** - defines which variables cause which
2. **Structural equations** - mathematical functions that determine each variable's value

For example, if `X` causes `Y`, the structural equation might be:
```
Y = f(X) + noise
```

Where `f` is some function (linear, quadratic, etc.) and `noise` is random variation.

---

## 2. Module-by-Module Breakdown

### 2.1. `noise.py` - Noise Distributions

This module defines the random noise added to each variable.

**Base class:**
```python
class NoiseGenerator:
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError
```

**Available distributions:**

| Class | Description | Key Parameters |
|-------|-------------|----------------|
| `GaussianNoise` | Normal distribution (bell curve) | `mean`, `std` |
| `UniformNoise` | Equal probability in range | `low`, `high` |
| `LaplacianNoise` | Heavier tails than Gaussian | `loc`, `scale` |
| `StudentTNoise` | Controllable heavy tails | `df` (degrees of freedom), `scale` |
| `MixtureNoise` | Combine multiple distributions | `components`, `weights` |

**How `MixtureNoise` works** (noise.py:58-74):
```python
def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
    # Randomly choose which component each sample comes from
    component_indices = rng.choice(len(self.components), size=n, p=weights)
    # Sample from each component for the assigned samples
    for i, comp in enumerate(self.components):
        mask = component_indices == i
        result[mask] = comp.sample(mask.sum(), rng)
```
This creates multi-modal distributions (e.g., data from two different populations).

---

### 2.2. `transforms.py` - Non-linear Transformations

Transforms define how a parent variable influences a child variable beyond simple linear relationships.

**Base class:**
```python
class Transform:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
```

**Available transforms:**

| Class | Formula | Use Case |
|-------|---------|----------|
| `IdentityTransform` | `x` | Linear relationships |
| `PolynomialTransform` | `sum(coef * x^deg)` | Quadratic, cubic effects |
| `SigmoidTransform` | `1 / (1 + exp(-scale*x))` | S-curve, bounded outputs |
| `TanhTransform` | `tanh(scale*x)` | Bounded to [-1, 1] |
| `SinusoidalTransform` | `amp * sin(freq*x + phase)` | Periodic relationships |
| `ExponentialTransform` | `scale * exp(rate*x)` | Exponential growth |
| `LogTransform` | `scale * log(\|x\| + ε)` | Diminishing returns |
| `ReLUTransform` | `max(0, x)` | Threshold effects |
| `ThresholdTransform` | `above_value if x > threshold else below_value` | Step functions |
| `CompositeTransform` | Chain multiple transforms | Complex relationships |
| `CustomTransform` | User-defined function | Anything else |

**The factory function** (transforms.py:165-187):
```python
def get_transform(name: str, **kwargs) -> Transform:
```
Lets you use strings like `"quadratic"`, `"sigmoid"` instead of instantiating classes.

---

### 2.3. `dag.py` - DAG Structure

This is the heart of the package. It defines the causal graph.

#### `Edge` class (dag.py:15-25)
Represents a causal link from `source` to `target`:
```python
@dataclass
class Edge:
    source: str      # Parent node name
    target: str      # Child node name
    weight: float    # Linear coefficient
    transform: Transform  # Non-linear transformation

    def compute_contribution(self, parent_value: np.ndarray) -> np.ndarray:
        return self.weight * self.transform(parent_value)
```

#### `Node` class (dag.py:28-43)
Represents a variable:
```python
@dataclass
class Node:
    name: str
    noise: NoiseGenerator    # Random variation
    is_root: bool = True     # Has no parents?
    intervened: bool = False # Fixed by intervention?
    intervention_value: Optional[float] = None
```

#### `DAG` class (dag.py:46-376)

**Key methods:**

1. **`add_node()`** (dag.py:64-103) - Add a variable:
   ```python
   dag.add_node("X", noise_std=1.0)  # Gaussian with std=1
   dag.add_node("Y", noise_type="uniform", noise_params={"low": -1, "high": 1})
   ```

2. **`add_edge()`** (dag.py:105-154) - Add a causal relationship:
   ```python
   dag.add_edge("X", "Y", weight=0.5, transform="quadratic")
   ```
   - Auto-creates nodes if they don't exist
   - Checks for cycles (would make it not a DAG)

3. **`_compute_topological_order()`** (dag.py:156-186) - Kahn's algorithm:
   - Finds an ordering where all parents come before children
   - Essential for sampling (must compute parents before children)
   - Detects cycles

4. **`intervene()`** (dag.py:201-219) - Do-calculus:
   ```python
   dag.intervene("X", value=2.0)  # Set X=2, breaking parent influence
   ```
   This is for **causal inference** - asking "what happens if we *set* X to 2" vs "what happens when we *observe* X=2".

---

### 2.4. `generator.py` - Data Sampling

The `DataGenerator` class samples from the SCM.

**Core sampling logic** (generator.py:30-70):
```python
def sample(self, n: int, return_dict: bool = False):
    order = self.dag._compute_topological_order()
    data = {}

    for name in order:
        node = self.dag.nodes[name]

        if node.intervened:
            # Fixed value (do-calculus)
            data[name] = np.full(n, node.intervention_value)
        else:
            # Structural equation: sum of parent contributions + noise
            value = np.zeros(n)
            for edge in self.dag.edges.get(name, []):
                parent_value = data[edge.source]
                value += edge.compute_contribution(parent_value)
            value += node.noise.sample(n, self.rng)
            data[name] = value
```

**The structural equation for each node is:**
```
X = Σ(weight_i * transform_i(parent_i)) + noise
```

**`sample_interventional()`** (generator.py:72-106) - Temporary interventions:
```python
# Sample with do(X=2) without permanently modifying the DAG
data = generator.sample_interventional(n=100, interventions={"X": 2.0})
```
Saves the current state, applies interventions, samples, then restores state.

---

### 2.5. `classification.py` - Classification Data

This module specializes in generating **labeled datasets** for classification tasks.

#### `FeatureSpec` (classification.py:15-40)
Defines a single feature:
```python
@dataclass
class FeatureSpec:
    name: str
    # Generative mode: different means per class
    loc_by_class: tuple[float, float] = (0.0, 0.0)  # (class_0_mean, class_1_mean)
    # Causal mode: contribution to Y
    weight_to_y: float = 0.0
    transform_to_y: Optional[str] = None
    # Noise
    noise_std: float = 1.0
    # Dependencies on other features
    parents: list[str] = []
    parent_weights: list[float] = []
    parent_transforms: list[str] = []
    # Optional final transform
    output_transform: Optional[str] = None
```

#### Two Modes of Generation

**1. Generative Mode** (`Y → X`):
- Sample label first: `Y ~ Bernoulli(class_balance)`
- Generate features conditional on Y: `X_i = loc[Y] + f(parents) + noise`
- Guarantees exact class balance
- Good for well-separated classes

**2. Causal Mode** (`X → Y`):
- Generate features first from the DAG
- Compute `P(Y=1|X) = link_function(intercept + Σ weight_i * X_i)`
- Sample `Y ~ Bernoulli(P(Y=1|X))`
- Reflects true causal structure
- Class balance achieved via intercept calibration

#### Key Methods

**`_build_dag()`** (classification.py:133-150):
Converts feature specs into a DAG for ordering.

**`_calibrate_intercept()`** (classification.py:152-176):
Uses bisection search (`scipy.optimize.brentq`) to find the intercept that achieves the target class balance:
```python
def balance_error(intercept):
    probs = self._apply_link(linear_pred + intercept)
    return probs.mean() - self.class_balance  # Want this to be 0
```

**`_apply_link()`** (classification.py:243-253):
Converts linear predictor to probability:
- **Logistic**: `1 / (1 + exp(-x))` - most common
- **Probit**: `Φ(x)` (normal CDF) - for latent variable interpretation
- **Linear**: Just clip to [0, 1] - not recommended

**`generate_batch()`** (classification.py:255-315):
Main method to get data.

**`from_random()`** (classification.py:548-710):
Factory method to create random classification problems with specified properties:
```python
gen = ClassificationDataGenerator.from_random(
    n_features=10,
    n_informative=6,
    n_direct_to_y=3,
    connectivity=0.4,  # Feature dependencies
    class_balance=0.2,
    mode="causal"
)
```

---

### 2.6. `patterns.py` - Common DAG Structures

Utility functions to create well-known causal structures:

| Function | Structure | Use Case |
|----------|-----------|----------|
| `create_chain(n)` | X0 → X1 → ... → Xn | Sequential causation |
| `create_fork(n)` | Z → {X1, X2, ..., Xn} | Confounding |
| `create_collider(n)` | {X1, ..., Xn} → Y | Selection bias |
| `create_mediator()` | X → M → Y and X → Y | Direct vs indirect effects |
| `create_instrument()` | Z → X → Y with U → X, U → Y | Instrumental variables |
| `create_random_dag(n, p)` | Random edges with probability p | Exploration |

**Important**: `create_random_dag` only allows edges from lower to higher indices, ensuring acyclicity (patterns.py:249-254).

---

## 3. How Everything Fits Together

Here's the flow when you call `generator.sample(n)`:

```
1. Compute topological order of nodes
2. For each node (in order):
   a. If intervened → use fixed value
   b. Otherwise:
      i.   Start with zeros
      ii.  For each parent edge:
           - Get parent's already-computed values
           - Apply transform: transform(parent_values)
           - Multiply by weight
           - Add to running sum
      iii. Sample noise from node's NoiseGenerator
      iv.  Add noise to sum
      v.   Store as this node's values
3. Return as numpy array or dict
```

---

## 4. Example Walkthrough

```python
from datagenerator import DAG, DataGenerator

# Create DAG
dag = DAG()
dag.add_node("Z", noise_std=1.0)      # Confounder
dag.add_node("X", noise_std=0.5)      # Treatment
dag.add_node("Y", noise_std=0.5)      # Outcome

dag.add_edge("Z", "X", weight=0.8)            # Z causes X
dag.add_edge("Z", "Y", weight=0.6)            # Z causes Y (confounding!)
dag.add_edge("X", "Y", weight=1.0, transform="quadratic")  # X causes Y (non-linearly)

# Generate data
gen = DataGenerator(dag, seed=42)
data = gen.sample(1000, return_dict=True)
# data = {"Z": array(...), "X": array(...), "Y": array(...)}

# What actually happens for each sample:
# Z = noise_Z                           (root node)
# X = 0.8 * Z + noise_X                 (linear from Z)
# Y = 0.6 * Z + 1.0 * X² + noise_Y      (quadratic from X)
```

---

## 5. Classification Example

```python
from datagenerator import ClassificationDataGenerator, FeatureSpec

# Generative mode: Y determines feature distributions
gen = ClassificationDataGenerator(
    mode="generative",
    class_balance=0.3,  # 30% positive class
    feature_specs=[
        FeatureSpec("age", loc_by_class=(40.0, 55.0), noise_std=10.0),
        FeatureSpec("income", loc_by_class=(50000, 80000), noise_std=15000),
        FeatureSpec("score", parents=["age", "income"],
                    parent_weights=[0.5, 0.0001], noise_std=5.0),
    ],
    n_noise_features=2,
    seed=42
)

X, y = gen.generate_batch(1000)
# X.shape = (1000, 5)  # 3 informative + 2 noise features
# y.shape = (1000,)    # Binary labels
```

```python
# Causal mode: Features determine Y
gen = ClassificationDataGenerator(
    mode="causal",
    class_balance=0.3,
    feature_specs=[
        FeatureSpec("age", noise_std=10.0, weight_to_y=0.05),
        FeatureSpec("income", noise_std=15000, weight_to_y=0.00002),
        FeatureSpec("score", parents=["age", "income"],
                    parent_weights=[0.5, 0.0001], noise_std=5.0,
                    weight_to_y=0.1, transform_to_y="sigmoid"),
    ],
    n_noise_features=2,
    link_function="logistic",
    seed=42
)

X, y = gen.generate_batch(1000)
```

---

## 6. Interventions (Do-Calculus)

Interventions let you ask causal questions like "What would Y be if we *set* X=2?"

```python
from datagenerator import DAG, DataGenerator

dag = DAG()
dag.add_node("Z").add_node("X").add_node("Y")
dag.add_edge("Z", "X", weight=1.0)
dag.add_edge("Z", "Y", weight=0.5)
dag.add_edge("X", "Y", weight=2.0)

gen = DataGenerator(dag, seed=42)

# Observational data: P(Y | X=2)
# Here X=2 might be because Z is high
obs_data = gen.sample(1000, return_dict=True)

# Interventional data: P(Y | do(X=2))
# Here we force X=2 regardless of Z
int_data = gen.sample_interventional(1000, interventions={"X": 2.0}, return_dict=True)

# The difference reveals the causal effect of X on Y
# (without confounding from Z)
```

---

## 7. Deep Dive: Topological Ordering and Cycle Detection

### What is Topological Order?

A **topological order** is a sequence of nodes where every parent appears before its children. This is essential for sampling because you can't compute a child's value until you know its parents' values.

Example DAG:
```
    Z
   / \
  v   v
  X   Y
   \ /
    v
    W
```

Valid topological orders:
- `[Z, X, Y, W]`
- `[Z, Y, X, W]`

Invalid (W before its parents):
- `[Z, W, X, Y]`

### Kahn's Algorithm

The code uses **Kahn's algorithm** (dag.py:156-186):

```python
def _compute_topological_order(self) -> list[str]:
    # Step 1: Count incoming edges (parents) for each node
    in_degree = {name: 0 for name in self.nodes}
    for target, edges in self.edges.items():
        in_degree[target] = len(edges)

    # Step 2: Start with root nodes (no parents, in_degree = 0)
    queue = [name for name, deg in in_degree.items() if deg == 0]
    order = []

    # Step 3: Process nodes one by one
    while queue:
        node = queue.pop(0)       # Take a node with no unprocessed parents
        order.append(node)         # Add to result

        # "Remove" this node by decrementing in_degree of its children
        for target, edges in self.edges.items():
            for edge in edges:
                if edge.source == node:
                    in_degree[target] -= 1
                    if in_degree[target] == 0:   # All parents now processed
                        queue.append(target)

    # Step 4: Cycle detection
    if len(order) != len(self.nodes):
        raise ValueError("Graph contains a cycle!")

    return order
```

### Visual Walkthrough

```
DAG:  Z → X → W
      Z → Y → W

Initial state:
  in_degree = {Z: 0, X: 1, Y: 1, W: 2}
  queue = [Z]           (only Z has no parents)
  order = []

Iteration 1: process Z
  order = [Z]
  Z's children: X, Y    (decrement their in_degree)
  in_degree = {Z: 0, X: 0, Y: 0, W: 2}
  queue = [X, Y]        (both now have in_degree 0)

Iteration 2: process X
  order = [Z, X]
  X's children: W
  in_degree = {Z: 0, X: 0, Y: 0, W: 1}
  queue = [Y]           (W still has 1 unprocessed parent)

Iteration 3: process Y
  order = [Z, X, Y]
  Y's children: W
  in_degree = {Z: 0, X: 0, Y: 0, W: 0}
  queue = [W]           (W now has all parents processed)

Iteration 4: process W
  order = [Z, X, Y, W]
  queue = []

Done! len(order) == len(nodes) ✓
```

### How Cycle Detection Works

If there's a cycle, some nodes will **never** reach `in_degree = 0` because they're waiting on each other.

```
Cycle:  A → B → C → A

Initial:
  in_degree = {A: 1, B: 1, C: 1}
  queue = []            (no node has in_degree 0!)
  order = []

While loop never runs because queue is empty.

len(order) = 0 ≠ len(nodes) = 3  →  CYCLE DETECTED!
```

Another example with a partial cycle:
```
    Z → A → B
        ↑   ↓
        └── C

in_degree = {Z: 0, A: 2, B: 1, C: 1}
queue = [Z]

Process Z:
  order = [Z]
  in_degree[A] decremented: {Z: 0, A: 1, B: 1, C: 1}
  queue = []            (A still waiting on C, which waits on B, which waits on A)

While loop ends.

len(order) = 1 ≠ len(nodes) = 4  →  CYCLE DETECTED!
```

The nodes in the cycle (A, B, C) never get added because they each depend on another node in the cycle.

### When is it Called?

1. **On `add_edge()`** - validates the new edge doesn't create a cycle:
   ```python
   def add_edge(self, source, target, ...):
       # ... add the edge ...
       try:
           self._compute_topological_order()
       except ValueError:
           self.edges[target].pop()  # Remove the bad edge
           raise ValueError(f"Adding edge {source} -> {target} would create a cycle")
   ```

2. **On `sample()`** - determines processing order:
   ```python
   def sample(self, n):
       order = self.dag._compute_topological_order()
       for name in order:
           # Process nodes in this order (parents before children)
   ```

The result is **cached** in `self._topological_order` and invalidated whenever the graph changes (node/edge added).

---

## 8. Deep Dive: Link Functions and Class Balance in Causal Mode

### The Challenge

In **causal mode**, you don't directly control class balance. The generation flow is:
1. Generate features X from the DAG
2. Compute P(Y=1|X) using a link function
3. Sample Y ~ Bernoulli(P(Y=1|X))

The proportion of Y=1 depends on the feature distributions, weights, link function, and intercept.

### What is a Link Function?

A **link function** maps a linear predictor to a probability:

```
η (linear predictor)  →  link function  →  P(Y=1) ∈ [0, 1]
     -∞ to +∞                                 0 to 1
```

The linear predictor is:
```
η = intercept + Σ(weight_i × transform_i(X_i))
```

Available link functions:

| Link | Formula | Notes |
|------|---------|-------|
| Logistic | `1 / (1 + e^(-η))` | Most common, nice mathematical properties |
| Probit | `Φ(η)` (normal CDF) | Assumes latent normal variable |
| Linear | `clip(η, 0, 1)` | Not recommended, can give exact 0 or 1 |

Visual representation of the logistic function:
```
P(Y=1)
  1 │                    ╭────────
    │                 ╭──╯
    │              ╭──╯
0.5 │─────────────●─────────────────  ← η=0 gives P=0.5
    │          ╭──╯
    │       ╭──╯
  0 │───────╯
    └──────────────────────────────→ η
         -4  -2   0   2   4
```

### How Intercept Calibration Works

The **intercept** shifts the probability curve up or down, controlling the baseline probability:

- **Negative intercept** → shifts curve right → lower average P(Y=1)
- **Positive intercept** → shifts curve left → higher average P(Y=1)

The `_calibrate_intercept()` method uses bisection search to find the intercept that achieves the target class balance:

```python
def _calibrate_intercept(self, n_samples=10000):
    # 1. Generate features with fixed seed
    features = self._generate_features(n_samples, temp_rng)

    # 2. Compute linear predictor WITHOUT intercept
    linear_pred = self._compute_linear_predictor(features, intercept=0.0)

    # 3. Find intercept where mean probability = target class balance
    def balance_error(intercept):
        probs = self._apply_link(linear_pred + intercept)
        return probs.mean() - self.class_balance

    # 4. Bisection search (brentq finds where balance_error = 0)
    self.intercept = brentq(balance_error, -20, 20)
```

### Important Trade-off: Class Balance vs. Exact Relationships

When you define a feature's relationship to Y:
```python
FeatureSpec("X", weight_to_y=2.0, transform_to_y="quadratic")
```

You might expect: `P(Y=1) = σ(2.0 × X²)`

But the actual equation is: `P(Y=1) = σ(intercept + 2.0 × X²)`

The **shape** of the relationship is preserved (still quadratic), but the **baseline probability shifts**.

Example with different intercepts:
```
                With intercept = 0          With intercept = -3
X = 0           P = σ(0) = 0.50             P = σ(-3) ≈ 0.05
X = 1           P = σ(2) ≈ 0.88             P = σ(-1) ≈ 0.27
X = 2           P = σ(8) ≈ 0.9997           P = σ(5)  ≈ 0.99
```

### Bypassing Auto-Calibration

If you want exact control over the X→Y relationship (at the cost of controlling class balance), set the intercept explicitly:

```python
gen = ClassificationDataGenerator(
    mode="causal",
    class_balance=0.3,      # Will be ignored
    intercept=0.0,          # Explicit - no auto-calibration
    feature_specs=[
        FeatureSpec("X", weight_to_y=2.0, transform_to_y="quadratic")
    ]
)
```

Now `P(Y=1) = σ(2.0 × X²)` exactly as defined, but class balance will be whatever the features produce.

### Why It's an Approximation

The calibration is approximate because:
1. **Finite sample calibration**: Uses 10,000 samples to estimate intercept
2. **Sampling variance**: `Y ~ Bernoulli(P)` introduces randomness
3. **Different random states**: Calibration uses fixed seed, generation uses different RNG

For large samples (n > 1000), actual class balance will be very close to target.

---

## 10. API Reference

### DAG Class

| Method | Description |
|--------|-------------|
| `add_node(name, noise_std=1.0, noise_type="gaussian", ...)` | Add a variable |
| `add_edge(source, target, weight=1.0, transform=None)` | Add causal relationship |
| `get_parents(node)` | Get parent node names |
| `get_children(node)` | Get child node names |
| `intervene(node, value)` | Set intervention |
| `clear_interventions()` | Remove all interventions |
| `copy()` | Deep copy the DAG |
| `describe()` | Text description |
| `plot()` | Matplotlib visualization |
| `to_ascii()` | ASCII representation |

### DataGenerator Class

| Method | Description |
|--------|-------------|
| `sample(n, return_dict=False)` | Generate n samples |
| `sample_interventional(n, interventions, return_dict=False)` | Sample with do() |
| `get_column_names()` | Column names in topological order |
| `to_dataframe(n)` | Sample as pandas DataFrame |

### ClassificationDataGenerator Class

| Method | Description |
|--------|-------------|
| `generate_batch(n, return_dataframe=False)` | Generate (X, y) tuple |
| `get_feature_names()` | All feature names |
| `get_full_dag()` | DAG including Y node |
| `plot_dag()` | Visualize with Y node |
| `describe()` | Text description |
| `from_random(...)` | Factory for random structure |

### Pattern Functions

| Function | Description |
|----------|-------------|
| `create_chain(n_nodes, ...)` | X0 → X1 → ... → Xn |
| `create_fork(n_children, ...)` | Z → {X1, ..., Xn} |
| `create_collider(n_parents, ...)` | {X1, ..., Xn} → Y |
| `create_mediator(...)` | X → M → Y, X → Y |
| `create_instrument(...)` | IV structure |
| `create_random_dag(n_nodes, edge_probability, ...)` | Random DAG |
