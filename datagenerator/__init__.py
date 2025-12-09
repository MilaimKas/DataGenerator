"""
DataGenerator - Synthetic data generation using DAG-based structural causal models.

Example usage:
    >>> from datagenerator import DAG, DataGenerator
    >>> dag = DAG()
    >>> dag.add_node("X", noise_std=1.0)
    >>> dag.add_node("Z", noise_std=0.5)  # Confounder
    >>> dag.add_node("Y", noise_std=0.5)
    >>> dag.add_edge("Z", "X", weight=0.8)
    >>> dag.add_edge("Z", "Y", weight=0.6)
    >>> dag.add_edge("X", "Y", weight=1.0, transform="quadratic")
    >>> generator = DataGenerator(dag)
    >>> data = generator.sample(n=1000)
"""

from .noise import (
    NoiseGenerator,
    GaussianNoise,
    UniformNoise,
    LaplacianNoise,
    StudentTNoise,
    MixtureNoise,
)

from .transforms import (
    Transform,
    IdentityTransform,
    PolynomialTransform,
    SigmoidTransform,
    TanhTransform,
    SinusoidalTransform,
    ExponentialTransform,
    LogTransform,
    ReLUTransform,
    ThresholdTransform,
    CompositeTransform,
    CustomTransform,
    get_transform,
)

from .dag import (
    Edge,
    Node,
    DAG,
)

from .generator import (
    DataGenerator,
)

from .classification import (
    FeatureSpec,
    ClassificationDataGenerator,
)

from .patterns import (
    create_chain,
    create_fork,
    create_collider,
    create_mediator,
    create_instrument,
    create_random_dag,
)

__all__ = [
    # Noise
    "NoiseGenerator",
    "GaussianNoise",
    "UniformNoise",
    "LaplacianNoise",
    "StudentTNoise",
    "MixtureNoise",
    # Transforms
    "Transform",
    "IdentityTransform",
    "PolynomialTransform",
    "SigmoidTransform",
    "TanhTransform",
    "SinusoidalTransform",
    "ExponentialTransform",
    "LogTransform",
    "ReLUTransform",
    "ThresholdTransform",
    "CompositeTransform",
    "CustomTransform",
    "get_transform",
    # DAG
    "Edge",
    "Node",
    "DAG",
    # Generators
    "DataGenerator",
    "FeatureSpec",
    "ClassificationDataGenerator",
    # Patterns
    "create_chain",
    "create_fork",
    "create_collider",
    "create_mediator",
    "create_instrument",
    "create_random_dag",
]
