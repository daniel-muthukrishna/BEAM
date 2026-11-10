"""
Score/Flow Matching loss and sampling structure 
"""
from typing import Tuple, List, Optional
import math
import numpy as np
import torch
import torch.nn as nn
from .probabilitypath import ProbabilityPath
from .simulator import Simulator, ODE, SDE, ODEIntegrator, EulerMaruyama, guided_output


class ScoreMatch(nn.Module):
    def __init__(self, nn_model: nn.Module, probability_path: ProbabilityPath,
                 device: torch.device, drop_prob: float = 0.1, epsilon: float = 1e-5,
                 architecture: str = "flow",
                 t_sampler: str = "uniform",
                 logit_normal_mean: float = 0.0,
                 logit_normal_std: float = 1.0,
                 low_t_boost_prob: float = 0.0,
                 low_t_boost_min: float = 0.001,
                 low_t_boost_max: float = 0.1,
                 sigma_min: float = 0.002):
        super().__init__()
        self.nn_model = nn_model.to(device)
        self.probability_path = probability_path
        self.drop_prob = drop_prob
        self.device = device
        self.epsilon = epsilon
        self.architecture = architecture
        assert t_sampler in ("uniform", "logit_normal", "uniform_sigma",
                             "theta_uniform"), \
            f"t_sampler must be 'uniform', 'logit_normal', 'uniform_sigma' " \
            f"or 'theta_uniform', got {t_sampler}"
        self.t_sampler = t_sampler
        self.sigma_min = sigma_min
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std
        assert 0.0 <= low_t_boost_prob < 1.0
        self.low_t_boost_prob = low_t_boost_prob
        self.low_t_boost_min = low_t_boost_min
        self.low_t_boost_max = low_t_boost_max
        self.ema = None

    def _sample_t(self, B: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample training times according to the configured sampler."""
        if self.t_sampler == "logit_normal":
            u = torch.randn(B, 1, 1, 1, device=self.device, generator=generator)
            t = torch.sigmoid(u * self.logit_normal_std + self.logit_normal_mean)
            t = t.clamp(self.epsilon, 1.0 - self.epsilon)
        elif self.t_sampler == "uniform_sigma":
            u = torch.rand(B, 1, 1, 1, device=self.device, generator=generator)
            sigma = self.sigma_min + (1.0 - self.sigma_min) * u
            t = torch.sqrt(1.0 - sigma ** 2).clamp_min(self.epsilon)
        elif self.t_sampler == "theta_uniform":
            th_lo = math.asin(self.epsilon)
            th_hi = math.acos(self.sigma_min)
            u = torch.rand(B, 1, 1, 1, device=self.device, generator=generator)
            t = torch.sin(th_lo + (th_hi - th_lo) * u)
        else:
            # 'uniform': identical to the old behavior, t in (epsilon, 1]
            t = torch.rand(B, 1, 1, 1, device=self.device, generator=generator) \
                * (1.0 - self.epsilon) + self.epsilon

        if self.low_t_boost_prob > 0.0:
            lo = math.log(max(self.low_t_boost_min, self.epsilon))
            hi = math.log(self.low_t_boost_max)
            u = torch.rand(B, 1, 1, 1, device=self.device, generator=generator)
            t_low = torch.exp(lo + (hi - lo) * u)
            use_low = torch.rand(B, 1, 1, 1, device=self.device, generator=generator) \
                < self.low_t_boost_prob
            t = torch.where(use_low, t_low, t)
        return t

    def forward(self, x: torch.Tensor, c: torch.Tensor, cam: torch.Tensor = None,
                t: torch.Tensor = None, noise: torch.Tensor = None,
                context_mask: torch.Tensor = None,
                generator: Optional[torch.Generator] = None,
                reduction: str = "mean"):
        """
        Args:
            x:            (B,1,H,W) input image.
            c:            (B,C) conditioning vector.
            cam:          (B,) camera indices.
            t:            (B,1,1,1) fixed times; sampled internally if None.
            noise:        fixed epsilon of x's shape; sampled internally if None.
            context_mask: (B,1,C) fixed CFG drop mask (1 = drop); Bernoulli
                          sampled internally if None. Pass zeros for a purely
                          conditional (deterministic) validation pass.
            generator:    torch.Generator for any internal sampling — lets the
                          trainer make validation reproducible.
            reduction:    'mean' returns a scalar loss;
                          'none' returns (per_sample_loss (B,), t (B,)) for
                          per-t-bin diagnostics.
        """
        B, _, C = c.shape

        if t is None:
            t = self._sample_t(B, generator)

        
        alpha_t = self.probability_path.alpha(t)
        beta_t = self.probability_path.beta(t)
        eps = noise if noise is not None else \
            torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        x_t = alpha_t * x + beta_t * eps

        if context_mask is None:
            drop = torch.bernoulli(
                torch.full((B, 1, 1), self.drop_prob, device=self.device, dtype=c.dtype),
                generator=generator,
            )
            context_mask = drop.expand(B, 1, C)

        # network
        pred = self.nn_model(x_t, c, t, context_mask, cam)

        # parametrizations 
        if self.architecture == "flow":
            # Standard velocity target, da*x + db*e. For the OT schedule
            alpha_dot = self.probability_path.alpha.derivative(t)
            beta_dot = self.probability_path.beta.derivative(t)
            target = (alpha_dot * x + beta_dot * eps).detach()
            per_sample = torch.mean(torch.square(pred - target), dim=(1, 2, 3))

        elif self.architecture == "noise":
            # Epsilon prediction: target is the standard-normal noise itself.
            target = eps.detach()
            per_sample = torch.mean(torch.square(pred - target), dim=(1, 2, 3))

        elif self.architecture == "v":
            # v-prediction: https://arxiv.org/abs/2202.00512. v = alpha_t * eps - beta_t * x1.
            target = (alpha_t * eps - beta_t * x).detach()
            per_sample = torch.mean(torch.square(pred - target), dim=(1, 2, 3))

        elif self.architecture == "score":
            # Denoising score matching, beta-weighted 
            score_ref, beta_t_ref = self.probability_path.conditional_score(x_t, x, t)
            target_eps = (-score_ref * beta_t_ref).detach()  # = eps
            per_sample = torch.mean(torch.square(beta_t_ref * pred + target_eps), dim=(1, 2, 3))

        else:
            raise ValueError(
                f"Architecture {self.architecture} must be one of 'flow', 'noise', 'score', 'v'"
            )

        # per-sample output mode for per-t-bin validation diagnostics
        if reduction == "none":
            with torch.no_grad():
                if self.architecture == "flow":
                    x1_hat = x_t + (1.0 - t) * pred
                elif self.architecture == "noise":
                    x1_hat = (x_t - beta_t * pred) / alpha_t.clamp_min(1e-3)
                elif self.architecture == "v":
                    denom = (alpha_t ** 2 + beta_t ** 2).clamp_min(1e-12)
                    x1_hat = (alpha_t * x_t - beta_t * pred) / denom
                else:  # score
                    x1_hat = (x_t + beta_t ** 2 * pred) / alpha_t.clamp_min(1e-3)
                flux_err = (x1_hat.float().mean(dim=(1, 2, 3))
                            - x.float().mean(dim=(1, 2, 3))) ** 2
            return per_sample, t.view(-1), flux_err
        return torch.mean(per_sample)

    def simulate(
        self,
        c_i: torch.Tensor,
        n_sample: int,
        size: Tuple[int, ...],
        device: torch.device,
        simulator: Simulator = ODEIntegrator,
        num_steps: int = 3000,
        guidance_scale: float = 1.0,
        num_save: int = 1,
        epsilon: float = 0.0001,
        cam: torch.Tensor = None,
        schedule: str = "uniform_sigma",
        final_tweedie: bool = True,
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
            cam: (n_datapoint,) 0-based camera indices matching c_i, or None

        Returns:
            Tuple of (final samples, intermediate samples, timesteps))
        """
        x0 = torch.randn(c_i.shape[0] * n_sample, 1, *size, device=device)

        if self.architecture != "flow" and schedule == "uniform_sigma":
            sig_hi = math.sqrt(1.0 - epsilon ** 2)
            sig_lo = math.sqrt(1.0 - (1.0 - epsilon) ** 2)
            sig = torch.linspace(sig_hi, sig_lo, num_steps, device=device)
            t0 = torch.sqrt(1.0 - sig ** 2)
        else:
            t0 = torch.linspace(epsilon, 1.0 - epsilon, num_steps, device=device)
        t = t0.unsqueeze(0).expand(1, num_steps)  # (1, num_steps)

        # Repeat conditioning to one entry per sample before it is captured by
        # the ODE/SDE wrapper.
        c_i = c_i.repeat_interleave(n_sample, dim=0).to(device)
        if cam is not None:
            cam = cam.repeat_interleave(n_sample, dim=0).to(device)

        if self.architecture == "flow":
            ode = ODE(self.nn_model, guidance_scale, cam=cam)
            simulator = simulator(ode, t, num_save=num_save)
        elif self.architecture in ("noise", "score", "v"):
            sde = SDE(
                nn_model=self.nn_model,
                probability_path=self.probability_path,
                guidance_value=guidance_scale,
                architecture=self.architecture,
                cam=cam,
            )
            simulator = EulerMaruyama(sde, t, num_save=num_save)
        else:
            raise ValueError(
                f"Architecture {self.architecture} must be one of 'flow', 'noise', 'score', 'v'"
            )

        xout, timesteps = simulator.simulate(x0, c_i)
        final_samples = xout[-1, ...]
        # terminal Tweedie step
        if final_tweedie and self.architecture in ("noise", "score", "v"):
            with torch.no_grad():
                t_f = t0[-1].expand(final_samples.shape[0], 1, 1, 1)
                out = guided_output(self.nn_model, final_samples, c_i,
                                    t_f, guidance_scale, cam)
                final_samples, _ = sde._x1_and_eps(out, final_samples,
                                                   t0[-1].view(1, 1, 1, 1))
        intermediate_samples = xout.cpu().numpy()
        timesteps = timesteps.cpu().numpy().tolist()
        return final_samples, intermediate_samples, timesteps


class EMA:
    """
    Maintains an exponential moving average (shadow copy) of model params.

    Args
    ----
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
        """Call after every optimiser step."""
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
