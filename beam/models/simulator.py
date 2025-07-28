
"""
Defining ODE and SDE classes and their corresponding simulators.
"""


import torch
import torch.nn as nn
from beam.models.unet import ContextUnet
from beam.models.probabilitypath import GaussianProbabilityPath, VPBeta
from tqdm import tqdm
from typing import Optional, Callable
from abc import ABC, abstractmethod

from torchdiffeq import odeint

# Helper functions for converting between flow and score models for gaussian paths

def noise2score(noise_model: ContextUnet, path: GaussianProbabilityPath) -> Callable:
    """
    Args:
        noise_model: ContextUnet
        path: GaussianProbabilityPath
    Returns:
        score: Callable
    Given a score model trained to output noise, return the score
    """
    def score(x: torch.Tensor, c: Optional[torch.Tensor], t: torch.Tensor, context_mask: torch.Tensor):
        b_t = path.beta(t).clamp_min(1e-5)
        return -noise_model(x, c, t, context_mask)/b_t
    return score

def flow2score(flow_model: ContextUnet, path: GaussianProbabilityPath) -> Callable:
    """
    Args:
        flow_model: ContextUnet
        guidance_value: float
        alpha: Alpha
        beta: Beta
    Returns:
        score: Callable
    Given a flow model trained to learn a gaussian path, the model can be rewritten using the schedulers to output a score model.
    """
    assert isinstance(path, GaussianProbabilityPath), "Only GaussianProbabilityPath is supported"
    alpha = path.alpha
    beta = path.beta

    def score(x: torch.Tensor, c: Optional[torch.Tensor], t: torch.Tensor, context_mask: torch.Tensor):
        alpha_t = alpha(t)
        beta_t = beta(t)
        dt_alpha_t = alpha.derivative(t)
        dt_beta_t = beta.derivative(t)


        num = alpha_t * flow_model(x, c, t, context_mask) - dt_alpha_t * x
        den = beta_t ** 2 * dt_alpha_t - alpha_t * dt_beta_t * beta_t

        return num / den
    return score

def score2flow(score_model: ContextUnet, path: GaussianProbabilityPath) -> Callable:
    """
    Args:
        score_model: ContextUnet
        guidance_value: float
        alpha: Alpha
        beta: Beta  
    Returns:
        flow: Callable
    Given a score model trained to learn a gaussian path, the model can be rewritten using the schedulers to output a flow model.
    """
    assert isinstance(path, GaussianProbabilityPath), "Only GaussianProbabilityPath is supported"
    alpha = path.alpha
    beta = path.beta
    @torch.no_grad()
    def flow(x: torch.Tensor, c: Optional[torch.Tensor], t: torch.Tensor, context_mask: torch.Tensor):
        score = score_model(x, c, t, context_mask)
        if isinstance(beta, VPBeta):
            return 1/t*(score + x)
        else:
            t = t.clamp_max(0.9998)
            alpha_t = alpha(t)
            da = alpha.derivative(t)
            db = beta.derivative(t)
            beta_t = beta(t)
            return (beta_t**2 * da/alpha_t - db*beta_t) * score + da/alpha_t*x
    return flow

class ODE():
    """
    Wrapper for U-net to be used as an ODE which can be solved using odeint.
    """
    def __init__(self, nn_model: ContextUnet, guidance_value: float):
        self.nn_model = nn_model
        self.guidance_value = guidance_value
    
    def velocity_field(self,x: torch.Tensor, c: Optional[torch.Tensor], t: torch.Tensor,):
        """
        Args:
            x: torch.Tensor
            c: torch.Tensor
            t: torch.Tensor
        Returns:
            gfc_drift: torch.Tensor
        Returns the drift of the ODE according to classifier-free guidance
        """
        context_mask = torch.cat([torch.zeros_like(c), torch.ones_like(c)], dim=0)
        c = torch.cat([c, c], dim=0)
        x = torch.cat([x, x], dim=0)
        t = torch.cat([t, t], dim=0)

        batch_output = self.nn_model(x, c, t, context_mask)
        conditional_drift, unconditional_drift = batch_output.chunk(2, dim=0)
        gfc_drift = (1-self.guidance_value) * unconditional_drift + self.guidance_value * conditional_drift

        return gfc_drift

class SDE():
    """
    Wrapper for U-net to be used as an SDE which can be solved using SDE Solvers.
    """
    def __init__(self, drift_model: ContextUnet, score_model: ContextUnet, guidance_value: float, sigma: float):
        self.guidance_value = guidance_value
        self.flow_model = drift_model
        self.score_model = score_model
        self.sigma = sigma
    
    def drift(self,x: torch.Tensor, c: torch.Tensor, t: torch.Tensor):
        """
        Args:
            x: torch.Tensor
            c: torch.Tensor
            t: torch.Tensor
        Returns:
            gfc_drift: torch.Tensor
        Returns the drift of the SDE according to classifier-free guidance
        """
        sigma = self.sigma
        if not isinstance(self.sigma, float):
            sigma = self.sigma(t)
        context_mask = torch.cat([torch.zeros_like(c), torch.ones_like(c)], dim=0)
        c = torch.cat([c, c], dim=0)
        x = torch.cat([x, x], dim=0)
        t = torch.cat([t, t], dim=0)

        vf_output = self.flow_model(x, c, t, context_mask)
        conditional_vector_field, unconditional_vector_field = vf_output.chunk(2, dim=0)
        gfc_vector_field = (1-self.guidance_value) * unconditional_vector_field + self.guidance_value * conditional_vector_field

        score_output = self.score_model(x, c, t, context_mask)
        conditional_score, unconditional_score = score_output.chunk(2, dim=0)
        gfc_score = (1-self.guidance_value )* unconditional_score + self.guidance_value * conditional_score
        return gfc_vector_field + 0.5*sigma**2 * gfc_score
    
    def score(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor):
        """
        Args:
            x: torch.Tensor
            c: torch.Tensor
            t: torch.Tensor
        Returns:
            gfc_score: torch.Tensor
        Return the score component of the Drift
        """
        if c is not None:
            context_mask = torch.cat([torch.zeros_like(c), torch.ones_like(c)], dim=0)
            c = torch.cat([c, c], dim=0)
            x = torch.cat([x, x], dim=0)
            t = torch.cat([t, t], dim=0)
            score_output = self.score_model(x, c, t, context_mask)
            conditional_score, unconditional_score = score_output.chunk(2, dim=0)
            gfc_score = (1-self.guidance_value )* unconditional_score + self.guidance_value * conditional_score
        return gfc_score
    
    def velocity_field(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor):
        """
        Args:
            x: torch.Tensor
            c: torch.Tensor
            t: torch.Tensor
        Returns:
            gfc_vector_field: torch.Tensor
        Return the velocity field component of the Drift
        """
        if c is not None:
            context_mask = torch.cat([torch.zeros_like(c), torch.ones_like(c)], dim=0)
            c = torch.cat([c, c], dim=0)
            x = torch.cat([x, x], dim=0)
            t = torch.cat([t, t], dim=0)
            vf_output = self.flow_model(x, c, t, context_mask)  
            conditional_vector_field, unconditional_vector_field = vf_output.chunk(2, dim=0)
            gfc_vector_field = (1-self.guidance_value) * unconditional_vector_field + self.guidance_value * conditional_vector_field
        return gfc_vector_field
    
    def diffusion_coeff(self, x, t):
        sigma = self.sigma
        if not isinstance(self.sigma, float):
            sigma = self.sigma(t)
        return sigma * torch.ones_like(x)
    
    def probability_flow(self):
        """
        Returns:
            ode: ODE
        Returns the ODE that corresponds to the probability flow of the SDE.
        """
        return ODE(self.flow_model, self.guidance_value)
    

class Simulator(ABC):
    """
    Abstract class for all simulators.
    """

    @abstractmethod
    @torch.no_grad()
    def simulate(self, *args, **kwargs):
        pass

class Euler(Simulator):
    """
    Basic Integrator used for debugging. All final ODE generations should come from ODEIntegrator.
    """
    def __init__(self, ode: ODE, t: torch.Tensor):
        self.ode = ode
        self.num_steps = t.shape[1]
        self.t = t  #shape (1, num_steps)

    @torch.no_grad()
    def simulate(self, x0: torch.Tensor, c: torch.Tensor):
        ts = self.t
        x = x0.clone()
        for k in tqdm(range(self.num_steps - 1)):
            t = ts[:, k]
            h  = ts[:, k+1] - ts[:, k]
            h = h.view(-1, 1, 1, 1)
            x = x + self.ode.velocity_field(x, c, t[..., None, None, None]) * h
        return x

class ODEIntegrator(Simulator):
    """
    Wrapper for odeint, found at https://github.com/rtqichen/torchdiffeq, to make compatible with defined ODE class
    Odeint returns a full trajectory
    """
    def __init__(self, ode: ODE, t: torch.Tensor, method: str = "dopri5", atol: float = 1e-5, rtol: float = 1e-5, num_save: int = 1):
        self.ode = ode
        self.num_steps = t.shape[1]
        self.t = t  #shape (1, num_steps)   
        self.method = method
        self.atol = atol
        self.rtol = rtol
        self.num_save = num_save
    @torch.no_grad()    
    def simulate(self, x0: torch.Tensor, c: torch.Tensor):
        ts = self.t[0] #odeint expects timem as 1d tensor
        bs = x0.shape[0]
        x = x0.clone()
        def drift_helper(t, x):
            return self.ode.velocity_field(x, c, t.expand(bs, 1, 1, 1)) 
        x = odeint(drift_helper, x, ts, method=self.method, atol=self.atol, rtol=self.rtol)
        if self.num_save == 1:
            ret_idx = -1
        else:
            ret_idx = torch.linspace(0, self.num_steps-1, steps = self.num_save, device=x.device, dtype = torch.long)
        return x[ret_idx, ...], ts[ret_idx] #return last step
    
class EulerMaruyama(Simulator):
    """
    Stochastic Euler Integrator for SDEs.
    """
    def __init__(self, sde: SDE, t: torch.Tensor, num_save: int = 1):
        self.sde = sde
        self.num_steps = t.shape[1]
        self.t = t   #shape (1, num_steps)
        self.num_save = num_save
    @torch.no_grad()
    def simulate(self, x0: torch.Tensor, c: torch.Tensor):
        B, C, H, W = x0.shape
        ts = self.t.expand(B, -1)
        x = x0.clone()
        if self.num_save == 1:
            ret_idx = torch.tensor([self.num_steps - 1], device=x.device, dtype=torch.long)
        else:
            ret_idx = torch.linspace(0, self.num_steps-1, steps = self.num_save, device=x.device, dtype = torch.long)
        ret_idx.clamp_(0, self.num_steps-1)
        xs_saved = torch.empty(
        self.num_save, B, C, H, W,
        device=x.device,
        dtype=x.dtype, 
        )
        sv_ptr = 0
        next_save_idx = ret_idx[sv_ptr]
        for k in tqdm(range(self.num_steps - 1)): 
            t = ts[:, k]
            if k == next_save_idx:
                xs_saved[sv_ptr] = x
                sv_ptr += 1
                if sv_ptr < self.num_save:
                    next_save_idx = ret_idx[sv_ptr].item()
            #update x
            h  = ts[:, k+1] - ts[:, k]
            h = h.view(-1, 1, 1, 1)
            x = x + self.sde.drift(x, c, t[..., None, None, None]) * h + self.sde.diffusion_coeff(x, t[..., None, None, None]) * torch.randn_like(x) * torch.sqrt(h)
        if (self.num_steps - 1) == next_save_idx:
            xs_saved[-1] = x
        return xs_saved, self.t[0, ret_idx]
class PredictorCorrector(Simulator):
    """
    Uses Euler Maruyama as the update step and n steps of Langevin MCMC to correct the result. Step size
    is based on the target SNR as shown in https://arxiv.org/abs/2011.13456
    """
    def __init__(self, sde: SDE, t: torch.Tensor, num_corrections: int=3, snr: float=0.16):
        self.sde = sde
        self.num_steps = t.shape[1]
        self.num_corrections = num_corrections
        self.snr = snr
        self.t = t  #shape (1, num_steps)

    @torch.no_grad()
    def simulate(self, x0: torch.Tensor, c: torch.Tensor):
        ts = self.t
        x_pred = x0.clone()
        for k in tqdm(range(self.num_steps - 1)):
            t = ts[:, k]
            h  = ts[:, k+1] - ts[:, k]
            h = h.view(-1, 1, 1, 1)
            sigma = self.sde.diffusion_coeff(x_pred, t[..., None, None, None])
            x_pred = x_pred + self.sde.drift(x_pred, c, t[..., None, None, None]) * h + sigma * torch.randn_like(x_pred) * torch.sqrt(h)

            x_corr = x_pred
            for _ in range(self.num_corrections):
                score = self.sde.score(x_corr, c, t[..., None, None, None])
                lrad_norm2 = score.pow(2).mean((1, 2, 3), keepdim=True)
                #step_size based on target snr
                eps = (self.snr*sigma)**2 / (lrad_norm2 + 1e-5)
                #Langevin correction
                x_corr += 0.5*eps*sigma**2 * score + sigma * torch.sqrt(eps) * torch.randn_like(x_corr)
            x_pred = x_corr
        return x_pred


  