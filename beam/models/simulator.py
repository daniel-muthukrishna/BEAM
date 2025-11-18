
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
        x0, t0 = x, t
        # Clamp time to avoid division by zero and extreme values
        t0 = t.clamp(1e-3, 1.0 - 1e-3)
        beta = self.beta(t0).clamp_min(1e-12)

        if not isinstance(self.sigma, float):
            sigma = self.sigma(t)
        context_mask = torch.cat([torch.zeros_like(c), torch.ones_like(c)], dim=0)

        # c = torch.cat([c, c], dim=0)
        # x = torch.cat([x, x], dim=0)
        # t = torch.cat([t, t], dim=0)

        #v param:
        gfc_vector_field = self.velocity_field(x, c, t)
        gfc_score = self.score(x, c, t)

        # if self.mode == "score":
        #     ##FIX THIS 
        #     vf_output = self.flow_model(x, c, t, context_mask)
        #     conditional_vector_field, unconditional_vector_field = vf_output.chunk(2, dim=0)
        #     gfc_vector_field = (1-self.guidance_value) * unconditional_vector_field + self.guidance_value * conditional_vector_field

        #     score_output = self.score_model(x, c, t, context_mask)
        #     conditional_score, unconditional_score = score_output.chunk(2, dim=0)
        #     gfc_score = (1-self.guidance_value )* unconditional_score + self.guidance_value * conditional_score
        # if self.mode == "noise":
        #     noise_output = self.flow_model(x, c, t, context_mask)
        #     conditional_noise, unconditional_noise = noise_output.chunk(2, dim=0)
        #     gfc_noise = (1-self.guidance_value )* unconditional_noise + self.guidance_value * conditional_noise
        #     gfc_score = -gfc_noise / beta 
        #     # For noise models: convert noise to vector field
        #     # The vector field depends on the specific alpha/beta schedule
        #     if isinstance(self.beta, VPBeta):
        #         # VP schedule: alpha(t) = t, beta(t) = sqrt(1-t^2)
        #         # Vector field: (score + x)/t (t0 is already clamped)
        #         gfc_vector_field = (gfc_score + x0)/t0
        #     else:
        #         # OT schedule: alpha(t) = t, beta(t) = 1-t
        #         # Vector field: score * beta^2 * d_alpha/alpha - d_beta*beta + d_alpha/alpha * x
        #         alpha_t = self.alpha(t0)
        #         da = self.alpha.derivative(t0)
        #         db = self.beta.derivative(t0)
        #         gfc_vector_field = (beta**2 * da/alpha_t - db*beta) * gfc_score + da/alpha_t*x0
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
            # score_output = self.score_model(x, c, t, context_mask)
            # conditional_score, unconditional_score = score_output.chunk(2, dim=0)
            # gfc_score = (1-self.guidance_value )* unconditional_score + self.guidance_value * conditional_score
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
            x0 = x
            # Clamp time to avoid division by zero
            t0 = t.clamp(1e-3, 1.0 - 1e-3)
            beta = self.beta(t0).clamp_min(1e-12)
            context_mask = torch.cat([torch.zeros_like(c), torch.ones_like(c)], dim=0)
            c = torch.cat([c, c], dim=0)
            x = torch.cat([x, x], dim=0)
            t = torch.cat([t, t], dim=0)
            # vf_output = self.flow_model(x, c, t, context_mask)  
            # conditional_vector_field, unconditional_vector_field = vf_output.chunk(2, dim=0)
            # gfc_vector_field = (1-self.guidance_value) * unconditional_vector_field + self.guidance_value * conditional_vector_field
            # score = -gfc_vector_field/beta
            # gfc_vector_field = (score + x0)/t0
            # vf_output = self.flow_model(x, c, t, context_mask)  
            # conditional_vector_field, unconditional_vector_field = vf_output.chunk(2, dim=0)
            # gfc_vector_field = (1-self.guidance_value) * unconditional_vector_field + self.guidance_value * conditional_vector_field
            # score = -gfc_vector_field/beta
            # gfc_vector_field = (score + x0)/t0
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
            ret_idx = -1
        else:
            ret_idx = torch.linspace(0, self.num_steps-1, steps = self.num_save, device=x.device, dtype = torch.long)
        return x[ret_idx, ...], ts[ret_idx] #return last step
    
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

        
class PosteriorSDE():
    f"""
    Self contained class for sampling of prior + liklihood models
    y is 256x256 latent light
    x_obs is 4096x4096 observed (light+stars)
    U: y -> 4096x4096 (linear upsample); UT: 4096x4096 -> y (downsample)
    Stars: score on full-res residual via tiling + overlap-add
    Light: score on y (low-res), optionally with CFG
    Posterior score (low-res): s_L(y) - UT( s_N(x - U(y)) )
    """

    def __init__(self, star_score, light_score, t: torch.Tensor, x_obs: torch.Tensor, sigma, guidance_value: float, upscale_factor: int = 16,
                 tile_size: int = 256, stride: int = 256, num_save: int = 1, num_corrections: int=0, corr_step_size: float=0.01,
                 star_tile_microbatch: int = 160):
        self.num_steps = t.shape[1]
        self.t = t   #shape (1, num_steps)
        self.num_save = num_save
        self.star_score = star_score
        self.light_score = light_score
        self.sigma = sigma                          # float or callable sigma(t) = sqrt(1 - t^2)
        self.guidance_value = guidance_value
        self.x_obs = x_obs
        self.alpha = OTAlpha()
        self.beta  = VPBeta()

        self.tile_size = tile_size                  # 256
        self.stride    = stride     
        self.upscale_factor = upscale_factor   
        self.num_corrections = num_corrections              # 16
        self.corr_step_size = corr_step_size              # 16
        self.star_tile_microbatch = star_tile_microbatch

    def _sigma_t(self, t, ref):
        if isinstance(self.sigma, float):
            return torch.full_like(ref, self.sigma)
        return self.sigma(t)


    def U(self, y):
        #y: (B,C,256,256) -> (B,C,4096,4096)
        return F.interpolate(y, scale_factor=self.upscale_factor, mode="bilinear", align_corners=False)

    def UT(self, g_full):
        #area pooling "inverse" of U
        return F.interpolate(g_full, scale_factor=1.0/self.upscale_factor, mode="area")

    def v_reparam(self, x, c, t, context_mask, model="star", mode="score"):
        """
        V-parameterization to score, assumes unit variance probability path
        """
        if model == "star":
            v_param = self.star_score(x, c, t, context_mask)                  # (B,C,h,w)
        elif model == "light":
            v_out   = self.light_score(x, c, t, context_mask)                 # (2B,C,h,w) for CFG

            v_cond, v_uncond = v_out.chunk(2, dim=0)
            v_param = (1 - self.guidance_value) * v_uncond + self.guidance_value * v_cond
            t, _ = t.chunk(2, dim=0)
            x, _ = x.chunk(2, dim=0)
        else:
            raise ValueError("model must be 'star' or 'light'")
        alpha = self.alpha(t)
        beta  = self.beta(t).clamp_min(1e-12)
        if mode == "score":
            return -(alpha * v_param/beta +  x) 
        elif mode == "vector_field":
            return -v_param/beta
        else:
            raise ValueError("mode must be 'score' or 'vector_field'")

    
    def star_fullres_old(self, residual, c_zero, t, mode):
        """
        Slides window over star score/vector field to get full-res star score/vector field
        """

        B, C, H, W = residual.shape
        T = 256
        assert H == 4096 and W == 4096, "This simple tiler assumes 4096x4096."
        out = torch.empty_like(residual)

        for by in range(16):        # 0..15
            for bx in range(16):    # 0..15
                y0, x0 = by*T, bx*T
                tile = residual[:, :, y0:y0+T, x0:x0+T]                 # (B,C,256,256)
                g_tile = self.v_reparam(
                    tile, c=c_zero, t=t, context_mask=torch.zeros_like(c_zero), model="star", mode=mode
                )  # (B,C,256,256)
                out[:, :, y0:y0+T, x0:x0+T] = g_tile
        return out

    def star_fullres(self, residual, c_zero, t, mode):
        """
        Slides window over star score/vector field to get full-res star score/vector field.
        Tiles are processed in microbatches to fit GPU memory.
        """
        B, C, H, W = residual.shape
        T = self.tile_size  # 256
        assert H == 4096 and W == 4096, "tiler assumes 4096x4096"

        # Reshape into tiles preserving the same order as the nested by/bx loops
        tiles = residual.view(B, C, H // T, T, W // T, T)           # B, C, 16, T, 16, T
        tiles = tiles.permute(0, 2, 4, 1, 3, 5).contiguous()        # B, 16, 16, C, T, T
        tiles = tiles.view(-1, C, T, T)                            # (B*256, C, 256, 256)

        g_tile_chunks = []
        mb = self.star_tile_microbatch
        total_tiles = tiles.shape[0]
        for start in range(0, total_tiles, mb):
            end = min(start + mb, total_tiles)
            tile_mb = tiles[start:end]
            c_mb = c_zero.expand(tile_mb.shape[0], -1, -1)
            t_mb = t.expand(tile_mb.shape[0], *t.shape[1:])
            cm_mb = torch.zeros(tile_mb.shape[0], *c_zero.shape[1:], device=c_zero.device, dtype=c_zero.dtype)

            g_mb = self.v_reparam(tile_mb, c=c_mb, t=t_mb, context_mask=cm_mb, model="star", mode=mode)
            g_tile_chunks.append(g_mb)

        g_tiles = torch.cat(g_tile_chunks, dim=0)

        out = g_tiles.view(B, H // T, W // T, C, T, T).permute(0, 3, 1, 4, 2, 5)
        out = out.reshape(B, C, H, W)
        return out



    def score(self, y: torch.Tensor, c: torch.Tensor, t: torch.Tensor):
        """
        Returns low-res posterior score (B,C,256,256):
            s_post(y,t) = s_L(y,t|c) - UT( s_N( x_obs - U(y), t ) )
        """
        STARMEAN = 0.08837062429384838 
        STARSTD =  0.8176902707359797
        LIGHTMEAN = 0.01978608595999928
        LIGHTSTD =  0.08876927679677006
        B = y.shape[0]

        y_up = self.U(y) 
        residual = (self.x_obs - y_up) * STARSTD + STARMEAN    #(B,C,4096,4096)

    
        c_zero = torch.zeros(1,1,24).to(c.device)
        sN_full = self.star_fullres(residual, c_zero, t, mode="score")/STARSTD   #divide by STD to get score in original space

        sN_low = self.UT(sN_full) #(B,C,256,256)

        y = y * LIGHTSTD + LIGHTMEAN
        x_light = torch.cat([y, y], dim=0)
        c_light = torch.cat([c, c], dim=0)
        t_light = torch.cat([t, t], dim=0)
        cm_light = torch.cat([torch.zeros_like(c), torch.ones_like(c)], dim=0)
        sL_low = self.v_reparam(x_light, c_light, t_light, cm_light, model="light", mode="score")/LIGHTSTD   #divide by STD to get score in original space
        return sL_low - sN_low, sL_low, sN_low

    def vector_field(y, c, t, score):
        t = t.clamp(1e-3, 1.0 - 1e-3)
        return (score + y)/t       
        
    def vector_field_light(self, y: torch.Tensor, c: torch.Tensor, t: torch.Tensor):
        """
        Returns low-res posterior vector field (B,C,256,256):
        """
        x_light = torch.cat([y, y], dim=0)
        c_light = torch.cat([c, c], dim=0)
        t_light = torch.cat([t, t], dim=0)
        cm_light = torch.cat([torch.zeros_like(c), torch.ones_like(c)], dim=0)
        vL_low = self.v_reparam(x_light, c_light, t_light, cm_light, model="light", mode="vector_field")

        return vL_low
    def score_light(self, y: torch.Tensor, c: torch.Tensor, t: torch.Tensor):
        x_light = torch.cat([y, y], dim=0)
        c_light = torch.cat([c, c], dim=0)
        t_light = torch.cat([t, t], dim=0)
        cm_light = torch.cat([torch.zeros_like(c), torch.ones_like(c)], dim=0)
        sL_low = self.v_reparam(x_light, c_light, t_light, cm_light, model="light", mode="score")
        return sL_low

   

    @torch.no_grad()
    def langevin_step(self, y, c, t, h=1.0):
        sigma_t = self._sigma_t(t, ref=y)
        s_post  = self.score(y, c, t)   
        drift = (0.5) * (sigma_t ** 2) * s_post * h
        y = y + drift 
        y = y + (sigma_t * torch.randn_like(y)) * (h ** 0.5) #stochastic update
        return y

    @torch.no_grad()
    def simulate(self, c, y0):
        """
        Samples from the posterior using Euler Maruyama predictor with optional Langevin corrections
        """
        y = y0.clone()        

        B, C, H, W = y0.shape
        ts = self.t.expand(B, -1) #(B, num_steps)
        if self.num_save == 1:
            ret_idx = torch.tensor([self.num_steps - 1], device=y.device, dtype=torch.long)
        else:
            ret_idx = torch.linspace(0, self.num_steps-1, steps = self.num_save, device=y.device, dtype = torch.long)
        ret_idx.clamp_(0, self.num_steps-1)
        ys_saved = torch.empty(
        self.num_save, B, C, H, W,
        device=y.device,
        dtype=y.dtype, 
        )
        sv_ptr = 0
        next_save_idx = ret_idx[sv_ptr]
        for k in tqdm(range(self.num_steps - 1)): 
            t = ts[:, k]
            if k == next_save_idx:
                ys_saved[sv_ptr] = y
                sv_ptr += 1
                if sv_ptr < self.num_save:
                    next_save_idx = ret_idx[sv_ptr].item()
            #update x
            h  = ts[:, k+1] - ts[:, k]
            h = h.view(-1, 1, 1, 1)
            
            sL_low = self.score_light(y, c, t[..., None, None, None])
            sigma_t = self._sigma_t(t, ref=y)
            y = y +  (self.vector_field_light(y, c, t[..., None, None, None]) + sL_low*0.5*sigma_t**2) * h 
            y = y + torch.randn_like(y) * sigma_t * (h ** 0.5) #prediction step using prior model

        for _ in range(self.num_corrections):
            score,_,_ = self.score(y, c, t[..., None, None, None])
            y = y + 0.5*sigma_t**2 * score * self.corr_step_size + torch.randn_like(y) * sigma_t * (self.corr_step_size ** 0.5) #posterior correction

        if (self.num_steps - 1) == next_save_idx:
            ys_saved[-1] = y
        return ys_saved, self.t[0, ret_idx]




    
