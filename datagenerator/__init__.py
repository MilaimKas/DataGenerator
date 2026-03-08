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

from .classification import (
    ClassificationDataGenerator,
    FeatureSpec,
)
from .dag import (
    DAG,
    Edge,
    Node,
)
from .generator import (
    DataGenerator,
)
from .noise import (
    GaussianNoise,
    LaplacianNoise,
    MixtureNoise,
    NoiseGenerator,
    StudentTNoise,
    UniformNoise,
)
from .patterns import (
    create_chain,
    create_collider,
    create_fork,
    create_instrument,
    create_mediator,
    create_random_dag,
)
from .transforms import (
    CompositeTransform,
    CustomTransform,
    ExponentialTransform,
    IdentityTransform,
    LogTransform,
    PolynomialTransform,
    ReLUTransform,
    SigmoidTransform,
    SinusoidalTransform,
    TanhTransform,
    ThresholdTransform,
    Transform,
    get_transform,
)

__all__ = [
    "DAG",
    "ClassificationDataGenerator",
    "CompositeTransform",
    "CustomTransform",
    # Generators
    "DataGenerator",
    # DAG
    "Edge",
    "ExponentialTransform",
    "FeatureSpec",
    "GaussianNoise",
    "IdentityTransform",
    "LaplacianNoise",
    "LogTransform",
    "MixtureNoise",
    "Node",
    # Noise
    "NoiseGenerator",
    "PolynomialTransform",
    "ReLUTransform",
    "SigmoidTransform",
    "SinusoidalTransform",
    "StudentTNoise",
    "TanhTransform",
    "ThresholdTransform",
    # Transforms
    "Transform",
    "UniformNoise",
    # Patterns
    "create_chain",
    "create_collider",
    "create_fork",
    "create_instrument",
    "create_mediator",
    "create_random_dag",
    "get_transform",
]
