
"""
Defining ODE and SDE classes and their corresponding simulators.
"""


import torch
import torch.nn.functional as F
from beam.models.unet import ContextUnet
from beam.models.probabilitypath import GaussianProbabilityPath, VPBeta, OTAlpha
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

            # Clamp t to avoid division by zero
            t_safe = t.clamp(1e-3, 1.0 - 1e-3)
            return 1/t_safe*(score + x)
        else:
            t_safe = t.clamp(1e-3, 1.0 - 1e-3)
            alpha_t = alpha(t_safe)
            da = alpha.derivative(t_safe)
            db = beta.derivative(t_safe)
            beta_t = beta(t_safe)
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
        # Clamp time to avoid numerical issues
        t_clamped = t.clamp(1e-3, 1.0 - 1e-3)
        
        context_mask = torch.cat([torch.zeros_like(c), torch.ones_like(c)], dim=0)
        c = torch.cat([c, c], dim=0)
        x = torch.cat([x, x], dim=0)
        t_batch = torch.cat([t_clamped, t_clamped], dim=0)

        batch_output = self.nn_model(x, c, t_batch, context_mask)
        conditional_drift, unconditional_drift = batch_output.chunk(2, dim=0)
        gfc_drift = (1-self.guidance_value) * unconditional_drift + self.guidance_value * conditional_drift

        return gfc_drift

class SDE():
    """
    Wrapper for U-net to be used as an SDE which can be solved using SDE Solvers.
    """
    def __init__(self, drift_model: ContextUnet, score_model: ContextUnet, guidance_value: float, sigma, mode: str = "score", probability_path=None):
        self.guidance_value = guidance_value
        self.flow_model = drift_model
        self.score_model = score_model
        self.sigma = sigma
        self.mode = mode
        if probability_path is not None:
            self.beta = probability_path.beta
            self.alpha = probability_path.alpha
        else:
            # Default fallback
            self.beta = VPBeta()
            self.alpha = OTAlpha()
    
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
        sigma = self.sigma if isinstance(self.sigma, float) else self.sigma(t)

        # v param
        gfc_vector_field = self.velocity_field(x, c, t)
        gfc_score = self.score(x, c, t)

        # TODO: SDE "noise"/"score" mode — convert a noise/score-model output to the
        # vector field per schedule. Unfinished; generation currently uses the ODE path.
        return gfc_vector_field + 0.5*(sigma**2) * gfc_score
    
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
        x0 = x
        beta = self.beta(t).clamp_min(1e-12)
        alpha = self.alpha(t)
        if c is not None:
            context_mask = torch.cat([torch.zeros_like(c), torch.ones_like(c)], dim=0)
            c = torch.cat([c, c], dim=0)
            x = torch.cat([x, x], dim=0)
            t = torch.cat([t, t], dim=0)
            #v param:
            v_param_output = self.score_model(x, c, t, context_mask)
            conditional_v_param, unconditional_v_param = v_param_output.chunk(2, dim=0)
            gfc_v_param = (1-self.guidance_value )* unconditional_v_param + self.guidance_value * conditional_v_param
            gfc_score = -(alpha*gfc_v_param + beta*x0)/beta
        return gfc_score
    
    def velocity_field(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor,):
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
            # Clamp time to avoid division by zero
            t0 = t.clamp(1e-3, 1.0 - 1e-3)
            beta = self.beta(t0).clamp_min(1e-12)
            context_mask = torch.cat([torch.zeros_like(c), torch.ones_like(c)], dim=0)
            c = torch.cat([c, c], dim=0)
            x = torch.cat([x, x], dim=0)
            t = torch.cat([t, t], dim=0)
            #vparam
            vparam_output = self.flow_model(x, c, t, context_mask)
            conditional_vector_field, unconditional_vector_field = vparam_output.chunk(2, dim=0)
            v_out = (1-self.guidance_value) * unconditional_vector_field + self.guidance_value * conditional_vector_field
            gfc_vector_field = -v_out/beta
            
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
        x = odeint(drift_helper, x, ts, method=self.method, atol=self.atol, rtol=self.rtol, )
        if self.num_save == 1:
            ret_idx = torch.tensor([self.num_steps - 1], device=x.device, dtype=torch.long)
        else:
            ret_idx = torch.linspace(0, self.num_steps-1, steps = self.num_save, device=x.device, dtype = torch.long)
        return x[ret_idx, ...], ts[ret_idx]
    
class EulerMaruyama(Simulator):
    """
    Stochastic Euler Integrator for SDEs.
    """
    def __init__(self, sde: SDE, t: torch.Tensor, num_save: int = 1, num_corrections: int=0, corr_step_size: float=0.3):
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

        
class LikelihoodLangevin():
    """
    Likelihood-only Langevin sampler at full 4096x4096 resolution.

    State variable X is a light estimate (4096x4096).
    Likelihood: P(obs | X) = Q(obs - X) where Q is the star distribution.
    Langevin update uses grad_X log Q(obs - X) = -s_star(obs - X).

    The star score is evaluated tile-wise (16x16 grid of 256x256 tiles)
    using the v-parameterized star U-Net (unconditional, no CFG).
    """

    def __init__(self, star_score, x_obs: torch.Tensor, sigma,
                 star_mean: float, star_std: float,
                 tile_size: int = 256, num_steps: int = 100,
                 step_size: float = 0.01, num_save: int = 1,
                 star_tile_microbatch: int = 160,
                 context_len: int = 12,
                 save_start_step: int = 0,
                 lazy_start_step: Optional[int] = None,
                 lazy_interval: int = 1):
        """
        Args (new):
            save_start_step: Step index at which snapshot collection begins.
                Snapshots are placed evenly between this step and num_steps-1.
                Use this as a "burn-in" for single-chain averaging.
            lazy_start_step: Step index at which lazy score updates begin.
                If None (default), the score is recomputed every iteration.
            lazy_interval: Number of iterations between score recomputations
                once lazy updates are active (i.e. k >= lazy_start_step).
                A value of 1 is equivalent to disabling lazy updates.
        """
        self.star_score = star_score
        self.x_obs = x_obs
        self.sigma = sigma
        self.star_mean = star_mean
        self.star_std = star_std
        self.tile_size = tile_size
        self.num_steps = num_steps
        self.step_size = step_size
        self.num_save = num_save
        self.star_tile_microbatch = star_tile_microbatch
        self.context_len = context_len
        self.save_start_step = max(0, min(int(save_start_step), num_steps - 1))
        self.lazy_start_step = lazy_start_step
        self.lazy_interval = max(1, int(lazy_interval))
        self.alpha = OTAlpha()
        self.beta = VPBeta()

    def _sigma_t(self, t, ref):
        if isinstance(self.sigma, float):
            return torch.full_like(ref, self.sigma)
        return self.sigma(t)

    def _v_to_score(self, x, v_param, t):
        """Convert v-parameterization output to score for the star model."""
        alpha = self.alpha(t)
        beta = self.beta(t).clamp_min(1e-12)
        return -(alpha * v_param / beta + x)

    def star_fullres(self, residual, t):
        """
        Evaluate star score on full 4096x4096 residual via tiling.
        Returns the reassembled 4096x4096 score.
        """
        B, C, H, W = residual.shape
        T = self.tile_size
        assert H == 4096 and W == 4096, "tiler assumes 4096x4096"

        tiles = residual.view(B, C, H // T, T, W // T, T)
        tiles = tiles.permute(0, 2, 4, 1, 3, 5).contiguous()
        tiles = tiles.view(-1, C, T, T)

        c_zero = torch.zeros(1, 1, self.context_len, device=residual.device, dtype=residual.dtype)
        g_tile_chunks = []
        mb = self.star_tile_microbatch
        for start in range(0, tiles.shape[0], mb):
            end = min(start + mb, tiles.shape[0])
            tile_mb = tiles[start:end]
            n = tile_mb.shape[0]
            c_mb = c_zero.expand(n, -1, -1)
            t_mb = t.expand(n, *t.shape[1:])
            cm_mb = torch.zeros(n, *c_zero.shape[1:], device=residual.device, dtype=residual.dtype)

            v_param = self.star_score(tile_mb, c_mb, t_mb, cm_mb)
            score_mb = self._v_to_score(tile_mb, v_param, t_mb)
            g_tile_chunks.append(score_mb)

        g_tiles = torch.cat(g_tile_chunks, dim=0)
        out = g_tiles.view(B, H // T, W // T, C, T, T).permute(0, 3, 1, 4, 2, 5)
        return out.reshape(B, C, H, W)

    def likelihood_score(self, X, t):
        """
        Compute grad_X log P(obs | X) = grad_X log Q(obs - X) = -s_star(obs - X).

        Normalizes the residual into the star model's training domain,
        evaluates the tiled star score, then un-normalizes.
        """
        residual_raw = self.x_obs - X
        residual_norm = (residual_raw - self.star_mean) / self.star_std

        s_star = self.star_fullres(residual_norm, t)

        # s_star is d/d(residual_norm) log Q(residual_norm).
        # Chain rule through normalization: d/d(residual_raw) = s_star / star_std.
        # Chain rule through subtraction: d/dX = -d/d(residual_raw).
        return -s_star / self.star_std

    @torch.no_grad()
    def simulate(self, X0):
        """
        Run Langevin dynamics on the likelihood P(obs | X) = Q(obs - X).

        Args:
            X0: (B, 1, 4096, 4096) initial light estimate

        Returns:
            (Xs_saved, step_indices): saved snapshots and their step numbers
        """
        X = X0.clone()
        B, C, H, W = X0.shape
        h = float(self.step_size)

        # Snapshots are placed evenly between save_start_step and num_steps-1.
        # With save_start_step=0 this reproduces the previous behaviour.
        if self.num_save <= 1:
            save_at = {self.num_steps - 1}
        else:
            save_at = set(
                torch.linspace(
                    self.save_start_step,
                    self.num_steps - 1,
                    self.num_save,
                    dtype=torch.long,
                ).tolist()
            )

        Xs_saved = torch.empty(self.num_save, B, C, H, W, device=X.device, dtype=X.dtype)
        sv_ptr = 0

        # Fixed time near t=1 (refining a near-clean image, not denoising from noise)
        t_fixed = torch.tensor(0.99, device=X.device).view(1, 1, 1, 1)
        sigma_t = self._sigma_t(t_fixed, ref=X)

        cached_grad = None
        steps_since_refresh = 0
        for k in tqdm(range(self.num_steps)):
            if k in save_at and sv_ptr < self.num_save:
                Xs_saved[sv_ptr] = X
                sv_ptr += 1

            lazy_active = (
                self.lazy_start_step is not None and k >= self.lazy_start_step
            )
            if (
                cached_grad is None
                or not lazy_active
                or steps_since_refresh >= self.lazy_interval
            ):
                grad = self.likelihood_score(X, t_fixed)
                cached_grad = grad
                steps_since_refresh = 1
                refreshed = True
            else:
                grad = cached_grad
                steps_since_refresh += 1
                refreshed = False

            drift = 0.5 * sigma_t ** 2 * grad * h
            # noise = sigma_t * (h ** 0.5) * torch.randn_like(X)
            print(
                f"  step {k:4d} | |score|={grad.abs().mean().item():.3e} "
                f"| |drift|={drift.abs().mean().item():.3e} "
                # f"| |noise|={noise.abs().mean().item():.3e}"
                f" | score={'fresh' if refreshed else 'cached'}"
            )
            # X = X + drift  + noise 
            X = X + drift 
        if sv_ptr < self.num_save:
            Xs_saved[-1] = X

        step_indices = sorted(save_at)
        return Xs_saved, step_indices




    
