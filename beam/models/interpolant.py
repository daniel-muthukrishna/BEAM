"""
Defining Score/Flow Matching loss and sampling structure
Keeps copy of EMA weights for sampling
"""
from typing import Dict, Tuple, Optional, Union, List
import numpy as np
import torch
import torch.nn as nn
from .probabilitypath import ProbabilityPath, OTAlpha, OTBeta, VPBeta
from torchvision import datasets, transforms, utils
from .simulator import Simulator, ODE, SDE, ODEIntegrator, score2flow, flow2score, noise2score, EulerMaruyama
from einops import repeat

class ScoreMatch(nn.Module):
    def __init__(self, nn_model: nn.Module, probability_path: ProbabilityPath, device: torch.device, drop_prob: float = 0.1, epsilon: float = 1e-5, architecture: str = "flow"):
        super().__init__()
        self.nn_model = nn_model.to(device)
        self.probability_path = probability_path
        self.drop_prob = drop_prob
        self.device = device
        self.epsilon = epsilon
        self.architecture = architecture
        self.ema = None
        self.loss_fn = nn.MSELoss()
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        B, _, C = c.shape
        t = torch.rand(B, 1, 1, 1, device=self.device) * (1. - self.epsilon) + self.epsilon #(BATCH_SIZE, 1, 1, 1)
        
        #Forward
        x_t,eps = self.probability_path.sample_conditional(x, t)
        # print(f'BATCH SIZE: {B}')


        drop = torch.bernoulli(torch.full((B, 1, 1), self.drop_prob, device=self.device, dtype=c.dtype))
        context_mask = drop.expand(B, 1, C)
        


        # score matching obj
        if self.architecture == "score":
            score_pred = self.nn_model(x_t, c, t, context_mask) 
            score_ref, beta_t = self.probability_path.conditional_score(x_t, x, t) 
            score_ref, beta_t = score_ref.detach(), beta_t.detach()
            epsilon = -score_ref * beta_t #really -epsilon
            batch_losses = torch.sum(torch.square(beta_t * score_pred + epsilon), dim=(1,2,3)) #minus minus
        elif self.architecture == "noise":
            noise_pred = self.nn_model(x_t, c, t, context_mask)
            noise_ref = eps
            noise_ref = noise_ref.detach()
            # batch_losses = torch.sum(torch.square(noise_pred - noise_ref), dim=(1,2,3))
            batch_losses = self.loss_fn(noise_pred, noise_ref)

        #flow matching obj
        elif self.architecture == "flow":
            flow_pred = self.nn_model(x_t, c, t, context_mask) 
            # flow_ref, beta_t = self.probability_path.conditional_vector_field(x_t, x, t)
            flow_ref = x - eps 
            flow_ref = flow_ref.detach()
            # batch_losses = torch.sum(torch.square(flow_pred - flow_ref), dim=(1,2,3))
            batch_losses = self.loss_fn(flow_pred, flow_ref)
            
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
        num_steps: int=6000,
        guidance_scale: float=1.0,
        num_save: int=1,
        epsilon: float=0.0001,
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
       
        
        x0 = torch.randn(c_i.shape[0]*n_sample, 1, *size, device=device) # random noise of shape (n_sample*n_datapoint, 1, *size) 
        if self.architecture == "flow":
            num_steps = 1000
            ode = ODE(self.nn_model, guidance_scale)
            t0 = torch.linspace(epsilon, 1.0 - epsilon, num_steps, device=device)
            t = t0.unsqueeze(0).expand(1, num_steps) #(1, num_steps)
            simulator = simulator(ode, t, num_save=num_save)
        elif self.architecture == "score":
            t0 = torch.linspace(epsilon, 1.0, num_steps, device=device)
            t = t0.unsqueeze(0).expand(1, num_steps) #(1, num_steps)
            sde = SDE(drift_model=score2flow(self.nn_model, self.probability_path), score_model=self.nn_model, sigma=self.probability_path.beta, guidance_value=guidance_scale)
            ode =  sde.probability_flow()
            simulator = simulator(ode, t, num_save=num_save)
        elif self.architecture == "noise":
            num_steps = 1500
            t0 = torch.linspace(epsilon, 1.0 - epsilon, num_steps, device=device)
            t = t0.unsqueeze(0).expand(1, num_steps) #(1, num_steps)
            # score_model = noise2score(self.nn_model, self.probability_path)
            sde = SDE(drift_model=self.nn_model, score_model=self.nn_model, sigma=VPBeta(), guidance_value=guidance_scale, mode="noise")
            simulator = EulerMaruyama(sde, t, num_save=num_save)
        else:
            raise ValueError(f"Architecture {self.architecture} must be either 'score' or 'flow' or 'noise'")
        
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
        self.backup = None

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

    @torch.no_grad()
    def store(self, model):
        self.backup = [p.detach().clone() for p in model.parameters()]

    @torch.no_grad()
    def restore(self, model):
        for p, b in zip(model.parameters(), self.backup):
            p.data.copy_(b)
        self.backup = None

    # convenience helpers for checkpointing
    def state_dict(self):
        return dict(beta=self.beta, step=self.step, shadow=self.shadow)

    def load_state_dict(self, d):
        self.beta, self.step, self.shadow = d["beta"], d["step"], d["shadow"]

    def to(self, device: torch.device):
        for idx, _ in enumerate(self.shadow):
            self.shadow[idx] = self.shadow[idx].to(device)


