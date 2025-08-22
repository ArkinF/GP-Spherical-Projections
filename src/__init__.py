# Gaussian Process Hyperparameter Learning via Spherical Projections

# Import data generation functions
from .data_generation import (
    generate_toy_1d,
    generate_toy_2d,
    generate_piecewise_data,
    generate_piecewise_linear_data,
    generate_multiscale_data
)

__all__ = [
    'generate_toy_1d',
    'generate_toy_2d', 
    'generate_piecewise_data',
    'generate_piecewise_linear_data',
    'generate_multiscale_data'
]