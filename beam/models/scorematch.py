"""
Defining Score/Flow Matching loss and sampling structure
Keeps copy of EMA weights for sampling
"""
from typing import Dict, Tuple, Optional, Union, List
import numpy as np
import torch
import torch.nn as nn
from .probabilitypath import GaussianProbabilityPath, LinearAlpha, LinearBeta
from torchvision import datasets, transforms, utils
from .simulator import Simulator, ODE, SDE, ODEIntegrator

class ScoreMatch(nn.Module):
    def __init__(self, nn_model: nn.Module, probability_path: GaussianProbabilityPath, device: torch.device, drop_prob: float = 0.1, epsilon: float = 1e-5, architecture: str = "score"):
        super().__init__()
        self.nn_model = nn_model.to(device)
        self.probability_path = probability_path
        self.drop_prob = drop_prob
        self.device = device
        self.epsilon = epsilon
        self.architecture = architecture
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        t = torch.rand(B, 1, 1, 1, device=self.device) #(BATCH_SIZE, 1, 1, 1)
        t.clamp_(min=self.epsilon, max=1-self.epsilon)  
        x_t = self.probability_path.sample_conditional(x, t)
        

        context_mask = torch.bernoulli(torch.zeros_like(c) + self.drop_prob)

        # score matching obj
        if self.architecture == "score":
            score_pred = self.nn_model(x_t, c, t, context_mask) 
            score_ref, beta_t = self.probability_path.conditional_score(x_t, x, t) 
            score_ref, beta_t = score_ref.detach(), beta_t.detach()
            epsilon = -score_ref * beta_t #really -epsilon
            batch_losses = torch.sum(torch.square(beta_t * score_pred + epsilon), dim=(1,2,3)) #minus minus

        #flow matching obj
        if self.architecture == "flow":
            flow_pred = self.nn_model(x_t, c, t, context_mask)
            flow_ref, beta_t = self.probability_path.conditional_vector_field(x_t, x, t)
            flow_ref = flow_ref.detach()
            batch_losses = torch.sum(torch.square(flow_pred - flow_ref), dim=(1,2,3))*beta_t**2
        else:
            raise ValueError(f"Architecture {self.architecture} must be either 'score' or 'flow'")
        
        return torch.mean(batch_losses)

    def simulate(
        self, 
        c_i: torch.Tensor, 
        n_sample: int, 
        size: Tuple[int, ...], 
        device: torch.device,
        simulator: Simulator=ODEIntegrator,
        num_steps: int=1000,
        guidance_scale: float=1.0,
        num_save: int=3
    ) -> Tuple[torch.Tensor, np.ndarray, List[int]]:
        """
        Sample from the model with specific conditioning.
        
        Args:
            c_i: Conditioning information
            n_sample: Number of samples per conditioning vector
            size: Size of samples
            device: Device to generate on
            simulator: Simulator to use
            num_steps: Number of timesteps
            guidance_scale: Guidance scale
            num_save: Number of intermediate (including endpoints) samples to save along the trajectory
            
        Returns:
            Tuple of (final samples, intermediate samples, timesteps))
        """
        print(f'Beginning Generation of {n_sample} samples with {num_steps} timesteps')
        t0 = torch.linspace(0.0, 1.0, num_steps, device=device)
        t = t0.unsqueeze(0).expand(n_sample, num_steps) #(n_sample, num_steps)
        x0 = torch.randn(c_i.shape[0]*n_sample, 1, *size, device=device) # random noise of shape (n_sample*n_datapoint, 1, *size) 

        ode = ODE(self.nn_model, guidance_scale)
        simulator = simulator(ode, t, num_save=num_save)
        c_i = c_i.repeat_interleave(n_sample, dim=0).to(device) #(n_sample*n_datapoint, 1, *size)
        xout,timesteps = simulator.simulate(x0, c_i) #(n_sample, num_save, 1, *size)

        final_samples = xout[-1, ...] #(n_sample, 1, *size)
        intermediate_samples = xout[:, ...].cpu().numpy() #(n_sample, num_save, 1, *size)
        timesteps = timesteps.cpu().numpy().tolist() #(num_steps)
        return final_samples, intermediate_samples, timesteps
        

class EMA:
    """
    Maintains an exponential moving average (shadow copy) of model params.

    Args
    ----
    model : nn.Module        • network whose parameters we track
    beta  : float = 0.9999   • decay; higher = slower, smoother
    update_after_step : int  • optional warm-up before first EMA update
    """

    def __init__(self, model, beta=0.9999, update_after_step=0):
        self.beta   = beta
        self.step   = 0
        self.warmup = update_after_step

        self.shadow = [p.detach().clone() for p in model.parameters()]

    @torch.no_grad()
    def update(self, model):
        """Call *once* after every optimiser step."""
        self.step += 1
        if self.step < self.warmup:                # optional warm-up
            self.shadow = [p.detach().clone() for p in model.parameters()]
            return

        beta = self.beta
        for s, p in zip(self.shadow, model.parameters()):
            s.mul_(beta).add_(p, alpha=1.0 - beta)

    @torch.no_grad()
    def copy_to(self, model):
        """Load EMA weights into `model` (for evaluation / sampling)."""
        for s, p in zip(self.shadow, model.parameters()):
            p.data.copy_(s)

    # convenience helpers for checkpointing
    def state_dict(self):
        return dict(beta=self.beta, step=self.step, shadow=self.shadow)

    def load_state_dict(self, d):
        self.beta, self.step, self.shadow = d["beta"], d["step"], d["shadow"]



