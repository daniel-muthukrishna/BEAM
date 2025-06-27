"""
Defining interpolant of form xt = alpha(t) * x1 + beta(t) * x0 + gamma(t) * z;  t in [0,1]
Convention is x0 = epsilon = N(0, I), x1 = Target (data) distribution

Also contains sampling methods for the interpolant path.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional
import torch
import torch.nn as nn
import numpy as np


class Alpha(ABC):
    """
    Schedule for weight of target distribution 
    """
    def __init__(self):
        #alpha(t = 0) = 0 -> X0 is gaussian
        assert torch.allclose(self(torch.zeros(1,1,1,1)), torch.zeros(1,1,1,1))
        #alpha(t = 1) = 1 -> X1 is sampled from target (data) distribution
        assert torch.allclose(self(torch.ones(1,1,1,1)), torch.ones(1,1,1,1))
        
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        pass

    def derivative(self, t: torch.Tensor) -> torch.Tensor:
        pass

class Beta(ABC):
    """
    Schedule for weight of init distribution
    """
    def __init__(self):
        #beta(t = 0) = 1 -> X0 is gaussian
        assert torch.allclose(self(torch.zeros(1,1,1,1)), torch.ones(1,1,1,1))
        #beta(t = 1) = 0 -> X1 is sampled from target (data) distribution
        assert torch.allclose(self(torch.ones(1,1,1,1)), torch.zeros(1,1,1,1))

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        pass

    def derivative(self, t: torch.Tensor) -> torch.Tensor:
        pass

class Gamma(ABC):
    """
    Noise Scheduler to add Gaussian noise when initial and target distributions are complex (non-Gaussian).
    Not used when one of the distributions is Gaussian.
    """
    def __init__(self):
        #gamma(t = 0) = 0 -> X0 is gaussian
        assert torch.allclose(self(torch.zeros(1,1,1,1)), torch.zeros(1,1,1,1))
        #gamma(t = 1) = 0 -> X1 is sampled from target (data) distribution
        assert torch.allclose(self(torch.ones(1,1,1,1)), torch.zeros(1,1,1,1))

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        pass

    def derivative(self, t: torch.Tensor) -> torch.Tensor:
        pass

# Specific Schedules
class LinearAlpha(Alpha):
    """
    t.shape = (batch_size, 1, 1, 1)
    """
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        assert t.shape[1:] == (1,1,1)
        return t

class LinearBeta(Beta):
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        assert t.shape[1:] == (1,1,1)
        return 1 - t

class VPAlpha(Alpha):
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return t

class VPBeta(Beta):
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1-t**2)
    def derivative(self, t: torch.Tensor) -> torch.Tensor:
        return -t / torch.sqrt(1-t**2)

class ProbabilityPath(ABC, nn.Module):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def sample_conditional(self, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pass

# Conditional Probability Paths
class GaussianProbabilityPath(ProbabilityPath):
    def __init__(self, alpha: Alpha, beta: Beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def sample_conditional(self, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample from conditional distribution p_t(x | x1) = N(x; alpha(t) * x1, beta(t)**2 * I)

        Args: 
            x1: (bs, c, h, w)
            t: (bs, 1, 1, 1)

        Returns:
            x_t: (bs, c, h, w)
        """
        epsilon = torch.randn_like(x1)
        return self.alpha(t) * x1 + self.beta(t) * epsilon
    
    def conditional_vector_field(self, x: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|x1)

        Args: 
            x: (bs, c, h, w)
            x1: (bs, c, h, w)
            t: (bs, 1, 1, 1)

        Returns:
            vector_field: (bs, c, h, w)
            beta_t: (bs, 1, 1, 1)
        """ 
        alpha_t = self.alpha(t) 
        beta_t = self.beta(t) 
        alpha_dot = self.alpha.derivative(t) 
        beta_dot = self.beta.derivative(t) 

        return (alpha_dot - beta_dot / beta_t * alpha_t) * x1 + beta_dot / beta_t * x, beta_t

    def conditional_score(self, x: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional score: grad_x log p_t(x|x1) and returns beta_t for optional epsilon calculation

        Args: 
            x: (bs, c, h, w)
            x1: (bs, c, h, w)
            t: (bs, 1, 1, 1)

        Returns:
            score: (bs, c, h, w)
            beta_t: (bs, 1, 1, 1)
        """ 
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        return (x1 * alpha_t - x) / beta_t**2, beta_t
