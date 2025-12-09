# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DataGenerator is a Python package for generating synthetic data from Directed Acyclic Graphs (DAGs) based on structural causal models. It supports:

- **Causal data generation**: Define DAGs with nodes (variables) and edges (causal relationships), then sample from structural equations
- **Classification data generation**: Create synthetic classification datasets with controlled class imbalance, feature dependencies, and causal/generative modes
- **Interventions (do-calculus)**: Generate interventional data by fixing variables to specific values
- **Non-linear transformations**: Apply transforms (quadratic, sigmoid, tanh, etc.) to causal relationships

## Package Structure

```
datagenerator/
├── __init__.py          # Public API exports
├── noise.py             # NoiseGenerator hierarchy
├── transforms.py        # Transform classes + get_transform() factory
├── dag.py               # Edge, Node, DAG classes
├── generator.py         # DataGenerator class
├── classification.py    # FeatureSpec, ClassificationDataGenerator
└── patterns.py          # Utility functions for common DAG patterns
```

## Architecture

1. **Noise Distributions** (`noise.py`): `NoiseGenerator`, `GaussianNoise`, `UniformNoise`, `LaplacianNoise`, `StudentTNoise`, `MixtureNoise`

2. **Transformations** (`transforms.py`): `IdentityTransform`, `PolynomialTransform`, `SigmoidTransform`, `TanhTransform`, `SinusoidalTransform`, etc. Use `get_transform(name)` factory for convenience.

3. **DAG Structure** (`dag.py`): `DAG`, `Node`, `Edge` - Core graph representation with topological ordering, cycle detection, and intervention support

4. **Data Generation**:
   - `generator.py`: `DataGenerator` - Samples from DAG structural equations
   - `classification.py`: `ClassificationDataGenerator`, `FeatureSpec` - Specialized for classification with two modes:
     - `generative`: Y sampled first, then X|Y (class-conditional features)
     - `causal`: X sampled first, then Y|X via link function

5. **Utility Functions** (`patterns.py`): `create_chain()`, `create_fork()`, `create_collider()`, `create_mediator()`, `create_instrument()`, `create_random_dag()`

## Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Run in Python REPL
python -c "from datagenerator import DAG, DataGenerator; dag = DAG(); dag.add_node('X'); dag.add_node('Y'); dag.add_edge('X', 'Y', weight=1.0); gen = DataGenerator(dag); print(gen.sample(10))"
```

## Dependencies

- **Required**: `numpy`
- **Optional**: `matplotlib` (for `DAG.plot()`, `ClassificationDataGenerator.plot_dag()`), `pandas` (for `to_dataframe()`), `scipy` (for `ClassificationDataGenerator` intercept calibration and probit link)

## Key Patterns

- DAG methods return `self` for method chaining: `dag.add_node("X").add_node("Y").add_edge("X", "Y")`
- Transforms can be specified by name string or instance: `transform="quadratic"` or `transform=PolynomialTransform(degrees=[2])`
- Interventions are temporary with `sample_interventional()` or persistent with `dag.intervene()`
