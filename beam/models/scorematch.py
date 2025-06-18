from typing import Dict, Tuple, Optional, Union, List
import numpy as np
import torch
import torch.nn as nn
from .probabilitypath import GaussianProbabilityPath, LinearAlpha, LinearBeta
from torchvision import datasets, transforms, utils

class ScoreMatch(nn.Module):
    def __init__(self, nn_model: nn.Module, probability_path: GaussianProbabilityPath, device: torch.device, drop_prob: float = 0.1):
        super().__init__()
        self.nn_model = nn_model.to(device)
        self.probability_path = probability_path.to(device)
        self.loss_mse = nn.MSELoss(reduction='sum')
        self.drop_prob = drop_prob
        self.device = device
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        t = torch.rand(B, 1, 1, 1, device=self.device) #(BATCH_SIZE, 1, 1, 1)
        x_t = self.probability_path.sample_conditional(x, t)

        context_mask = torch.bernoulli(torch.zeros_like(c) + self.drop_prob)

        # score matching obj
        score_pred = self.nn_model(x_t, c, t, context_mask) 
        score_ref, beta_t = self.probability_path.conditional_score(x_t, x, t) 
        score_ref, beta_t = score_ref.detach(), beta_t.detach()
        epsilon = -score_ref * beta_t #really -epsilon
        batch_losses = torch.sum(torch.square(beta_t * score_pred + epsilon), dim=(1,2,3)) #minus minus

        #flow matching obj
        # flow_pred = self.nn_model(x_t, c, t, context_mask)
        # flow_ref = self.probability_path.conditional_vector_field(x_t, x, t)
        # flow_ref = flow_ref.detach()
        # batch_losses = torch.sum(torch.square(flow_pred - flow_ref), dim=(1,2,3))

        return torch.mean(batch_losses)

    def simulate(*args):
        #OPTIONS
        #ODE Something Better than Euler like RK4 or other adaptive solvers
        #SDE Euler Maruyama or Heun
        pass

