"""
Model components for BEAM.

This module provides diffusion model components and architectures.

These model scripts (unet.py and diffusion.py) define a conditional diffusion model for image training and generation.

The code is adapted from:
https://github.com/TeaPearce/Conditional_Diffusion_MNIST
which was originally modified from:
https://github.com/cloneofsimo/minDiffusion

Based on research from:
- DDPM: https://arxiv.org/abs/2006.11239
- Classifier-Free Diffusion Guidance: https://arxiv.org/abs/2207.12598
- ImageGen: https://arxiv.org/abs/2205.11487

"""

from beam.models.unet import (
    ContextUnet,
    ResidualConvBlock,
    UnetDown,
    UnetUp,
    EmbedFC
)
from beam.models.scorematch import ScoreMatch, EMA
from beam.models.probabilitypath import GaussianProbabilityPath, LinearAlpha, LinearBeta
from beam.models.simulator import (
    ODE,
    SDE,
    ODEIntegrator,
    EulerMaruyama,
    PredictorCorrector,
)

__all__ = [
    'ContextUnet',
    'ResidualConvBlock',
    'UnetDown',
    'UnetUp',
    'EmbedFC',
    'ScoreMatch',
    'EMA',
    'GaussianProbabilityPath',
    'LinearAlpha',
    'LinearBeta',
    'ODE',
    'SDE',
    'ODEIntegrator',
    'EulerMaruyama',
    'PredictorCorrector',
]