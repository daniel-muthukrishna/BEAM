"""
Defining Score/Flow Matching loss and sampling structure
Keeps copy of EMA weights for sampling
"""
from typing import Dict, Tuple, Optional, Union, List
import numpy as np
import torch
import torch.nn as nn
from beam.models.simulator import ODEIntegrator

class SBD(nn.Module):
    def __init__(self, nn_model: nn.Module, sde, device: torch.device, drop_prob: float = 0.1, epsilon: float = 1e-5, architecture: str = "flow"):
        super().__init__()
        self.nn_model = nn_model.to(device)
        self.sde = sde
        self.drop_prob = drop_prob
        self.device = device
        self.epsilon = epsilon
        self.architecture = architecture
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        t = torch.rand(B, 1, 1, 1, device=self.device) * (1. - self.epsilon) + self.epsilon #(BATCH_SIZE, 1, 1, 1)
        x_t, eps, score = self.sde.forward(x, t)
        

        context_mask = torch.bernoulli(torch.zeros_like(c) + self.drop_prob)

        # score matching obj
        noise_pred = self.nn_model(x_t, c, t, context_mask) 
        batch_losses = torch.sum(torch.square(noise_pred - eps), dim=(1,2,3)) #minus minus
        
        return torch.mean(batch_losses)

    def simulate(
        self, 
        c_i: torch.Tensor, 
        n_sample: int, 
        size: Tuple[int, ...], 
        device: torch.device,
        simulator=ODEIntegrator,
        num_steps: int=1000,
        guidance_scale: float=1.0,
        num_save: int=3,
        epsilon: float=0.0001
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
       
        
        x0 = torch.randn(c_i.shape[0]*n_sample, 1, *size, device=device) * self.sde.sigma_max # random noise of shape (n_sample*n_datapoint, 1, *size) 
        ode = self.sde
        t0 = torch.linspace(1.0 - epsilon, epsilon, num_steps, device=device)
        t = t0.unsqueeze(0).expand(1, num_steps) #(1, num_steps)
        simulator = simulator(ode, t, num_save=num_save)
        
        c_i = c_i.repeat_interleave(n_sample, dim=0).to(device) #(n_sample*n_datapoint, 1, *size)
        xout, timesteps = simulator.simulate(x0, c_i) #(num_save, n_sample, 1, *size)
        final_samples = xout[-1, ...] #(n_sample, 1, *size)
        intermediate_samples = xout.cpu().numpy() #(num_save, n_sample, 1, *size)
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


class VESDE():
    def __init__(self, sigma_min: float, sigma_max: float, device: torch.device, model: nn.Module, guidance_value: float):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.device = device
        self.model = model
        self.guidance_value = guidance_value
    def diffusion(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        return sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                device=t.device))
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        noise = torch.randn_like(x)
        x_t = x + sigma * noise
        score = -noise / sigma
        return x_t, noise, score
    def reversePflow(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        g_t = self.diffusion(x, t)
        context_mask = torch.cat([torch.zeros_like(c), torch.ones_like(c)], dim=0)
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        c = torch.cat([c, c], dim=0)
        x = torch.cat([x, x], dim=0)
        t = torch.cat([t, t], dim=0)
        noise_output = self.model(x, c, t, context_mask)
        conditional_noise, unconditional_noise = noise_output.chunk(2, dim=0)
        gfc_noise = (1-self.guidance_value )* unconditional_noise + self.guidance_value * conditional_noise
        score = -gfc_noise / sigma
        return -0.5 * (g_t ** 2)*score
    
    