"""
Defining Score/Flow Matching loss and sampling structure
Keeps copy of EMA weights for sampling
"""
from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn
from .probabilitypath import ProbabilityPath
from .simulator import Simulator, ODE, SDE, ODEIntegrator, EulerMaruyama

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

    def forward(self, x: torch.Tensor, c: torch.Tensor, cam: torch.Tensor = None) -> torch.Tensor:
        B, _, C = c.shape
        t = torch.rand(B, 1, 1, 1, device=self.device) * (1. - self.epsilon) + self.epsilon #(BATCH_SIZE, 1, 1, 1)
        
        #Forward
        x_t,eps, alpha_t, beta_t = self.probability_path.sample_conditional(x, t)

        drop = torch.bernoulli(torch.full((B, 1, 1), self.drop_prob, device=self.device, dtype=c.dtype))
        context_mask = drop.expand(B, 1, C)

        #network
        pred = self.nn_model(x_t, c, t, context_mask, cam)

        #parametrizations
        if self.architecture == "flow":
            # Standard velocity target, da*x + db*e. For the OT schedule (alpha_dot=1, beta_dot=-1) this is x1 - eps.
            alpha_dot = self.probability_path.alpha.derivative(t)
            beta_dot = self.probability_path.beta.derivative(t)
            target = (alpha_dot * x + beta_dot * eps).detach()
            batch_losses = self.loss_fn(pred, target)

        elif self.architecture == "noise":
            # Epsilon prediction: target is the standard-normal noise itself.
            target = eps.detach()
            batch_losses = self.loss_fn(pred, target)

        elif self.architecture == "v":
            # v-prediction: https://arxiv.org/abs/2202.00512. v = alpha_t * eps - beta_t * x1.
            target = (alpha_t * eps - beta_t * x).detach()
            batch_losses = self.loss_fn(pred, target)

        elif self.architecture == "score":
            # Denoising score matching, beta-weighted 
            score_ref, beta_t = self.probability_path.conditional_score(x_t, x, t)
            target_eps = (-score_ref * beta_t).detach()  # = eps
            batch_losses = torch.sum(torch.square(beta_t * pred + target_eps), dim=(1, 2, 3))

        else:
            raise ValueError(
                f"Architecture {self.architecture} must be one of 'flow', 'noise', 'score', 'v'"
            )

        return torch.mean(batch_losses)

    def simulate(
        self, 
        c_i: torch.Tensor, 
        n_sample: int, 
        size: Tuple[int, ...], 
        device: torch.device,
        simulator: Simulator=ODEIntegrator,
        num_steps: int=3000,
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
        t0 = torch.linspace(epsilon, 1.0 - epsilon, num_steps, device=device)
        t = t0.unsqueeze(0).expand(1, num_steps) #(1, num_steps)

        if self.architecture == "flow":
            ode = ODE(self.nn_model, guidance_scale)
            simulator = simulator(ode, t, num_save=num_save)
        elif self.architecture in ("noise", "score", "v"):
            sde = SDE(
                nn_model=self.nn_model,
                probability_path=self.probability_path,
                guidance_value=guidance_scale,
                architecture=self.architecture,
            )
            simulator = EulerMaruyama(sde, t, num_save=num_save)
        else:
            raise ValueError(
                f"Architecture {self.architecture} must be one of 'flow', 'noise', 'score', 'v'"
            )

        c_i = c_i.repeat_interleave(n_sample, dim=0).to(device) #(n_sample*n_datapoint, 1, *size)
        xout, timesteps = simulator.simulate(x0, c_i) #(num_save, n_sample, 1, *size)
        final_samples = xout[-1, ...] #(n_sample, 1, *size)
        intermediate_samples = xout.cpu().numpy() #(num_save, n_sample, 1, *size)
        timesteps = timesteps.cpu().numpy().tolist() #(num_steps)
        return final_samples, intermediate_samples, timesteps
        

class EMA:
    """
    Maintains an exponential moving average (shadow copy) of model params.

    Args:
        model : nn.Module       
        beta  : float = 0.9999   
        update_after_step : int  
    """

    def __init__(self, model, beta=0.9999, update_after_step=0):
        self.beta   = beta
        self.step   = 0
        self.warmup = update_after_step
        self.backup = None

        self.shadow = [p.detach().clone() for p in model.parameters()]

    @torch.no_grad()
    def update(self, model):
        """Called after every optimiser step."""
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


