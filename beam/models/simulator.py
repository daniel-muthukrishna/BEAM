import torch
import torch.nn as nn
from beam.models.unet import ContextUnet
from beam.models.probabilitypath import Alpha, Beta

from tqdm import tqdm

from abc import ABC, abstractmethod

# class Flow2Score(nn.Module):
#     def __init__(self, flow_model: ContextUnet, guidance_value: float, alpha: Alpha, beta: Beta):
#         super().__init__()
#         self.flow_model = flow_model
#         self.guidance_value = guidance_value
#         self.alpha = alpha
#         self.beta = beta
#     def forward(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor):
#         ut = self.vector_field(x,t) 
#         alpha_t = self.alpha(t) # (bs,)
#         da = self.alpha.dt(t)
#         db = self.beta.dt(t)
#         beta_t = self.beta(t) # (bs,)
#         return (alpha_t*ut - da*x) / (beta_t**2 * da - alpha_t*db*beta_t)




class ODE():
    def __init__(self, nn_model: ContextUnet, guidance_value: float):
        self.nn_model = nn_model
        self.guidance_value = guidance_value
    
    def drift(self,x: torch.Tensor, c: torch.Tensor, t: torch.Tensor,):
        if c is not None:
            context_mask = torch.cat([torch.zeros_like(c), torch.ones_like(c)], dim=0)
            c = torch.cat([c, c], dim=0)
            x = torch.cat([x, x], dim=0)
            t = torch.cat([t, t], dim=0)
            # assert c.shape[-1] == 12, f"Context must have 12 channels, has shape {c.shape}"

            batch_output = self.nn_model(x, c, t, context_mask)
            conditional_drift, unconditional_drift = batch_output.chunk(2, dim=0)
            gfc_drift = 1-self.guidance_value * unconditional_drift + self.guidance_value * conditional_drift
        # else:
        #     context_mask = torch.ones_like(x)
        #     c = torch.zeros_like(x)
        #     gfc_drift = self.nn_model(x, c, t, context_mask)
        return gfc_drift

class SDE():
    def __init__(self, drift_model: ContextUnet, score_model: ContextUnet, guidance_value: float, sigma: float):
        self.guidance_value = guidance_value
        self.flow_model = drift_model
        self.score_model = score_model
        self.sigma = sigma
    
    def drift(self,x: torch.Tensor, c: torch.Tensor, t: torch.Tensor):
        if c is not None:
            context_mask = torch.cat([torch.zeros_like(c), torch.ones_like(c)], dim=0)
            c = torch.cat([c, c], dim=0)
            x = torch.cat([x, x], dim=0)
            t = torch.cat([t, t], dim=0)
            # assert c.shape[-1] == 12, f"Context must have 12 channels, has shape {c.shape}"

            vf_output = self.flow_model(x, c, t, context_mask)
            conditional_vector_field, unconditional_vector_field = vf_output.chunk(2, dim=0)
            gfc_vector_field = 1-self.guidance_value * unconditional_vector_field + self.guidance_value * conditional_vector_field

            score_output = self.score_model(x, c, t, context_mask)
            conditional_score, unconditional_score = score_output.chunk(2, dim=0)
            gfc_score = 1-self.guidance_value * unconditional_score + self.guidance_value * conditional_score

        # else:
        #     context_mask = torch.ones_like(c)
        #     c = torch.zeros_like(x)
        #     gfc_vector_field = self.flow_model(x, c, t, context_mask)
        #     gfc_score = self.score_model(x, c, t, context_mask)

        return gfc_vector_field + 0.5*self.sigma**2 * gfc_score
    
    def diffusion_coeff(self, x):
        return self.sigma * torch.ones_like(x)
    
    def probability_flow(self):
        return ODE(self.flow_model, self.guidance_value)
    

class Simulator(ABC):
    @abstractmethod
    @torch.no_grad()
    def simulate(self, *args, **kwargs):
        pass

class Euler(Simulator):
    def __init__(self, ode: ODE, t: torch.Tensor):
        self.ode = ode
        self.num_steps = t.shape[1]
        self.t = t  #shape (B, num_steps)

    @torch.no_grad()
    def simulate(self, x0: torch.Tensor, c: torch.Tensor):
        ts = self.t
        x = x0.clone()
        for k in tqdm(range(self.num_steps - 1)):
            t = ts[:, k]
            h  = ts[:, k+1] - ts[:, k]
            h = h.view(-1, 1, 1, 1)
            x = x + self.ode.drift(x, c, t[..., None, None, None]) * h
        return x
    
class EulerMaruyama(Simulator):
    def __init__(self, sde: SDE, t: torch.Tensor):
        self.sde = sde
        self.num_steps = t.shape[1]
        self.t = t  #shape (B, num_steps)

    @torch.no_grad()
    def simulate(self, x0: torch.Tensor, c: torch.Tensor):
        ts = self.t
        x = x0.clone()
        for k in tqdm(range(self.num_steps - 1)):
            t = ts[:, k]
            h  = ts[:, k+1] - ts[:, k]
            h = h.view(-1, 1, 1, 1)
            x = x + self.sde.drift(x, c, t[..., None, None, None]) * h + self.sde.diffusion_coeff(x) * torch.randn_like(x) * torch.sqrt(h)
        return x


if __name__ == "__main__":
    device = torch.device("cuda")
    model = ContextUnet(
        in_channels=1,
        in_dim=10,
        n_feat=64,
    ).to(device)
    new_state_dict = {}
    for k, v in torch.load("/pdo/users/djtufto/score_real_match_test.pth", map_location="cpu").items():
        if k.startswith("nn_model."):
            new_state_dict[k[len("nn_model."):]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

    model.eval()

    from torchvision.utils import make_grid, save_image
    import matplotlib.pyplot as plt
    
    samples_per_class = 10
    n_classes = 10
    batch_size = samples_per_class * n_classes  # 100
    image_size = 32
    num_steps = 1000
    guidance_scales = [30.0]
    fig, axes = plt.subplots(1, len(guidance_scales), figsize=(10 * len(guidance_scales), 10))


    for idx, guidance_scale in enumerate(guidance_scales):
        ode = ODE(model, guidance_scale)
        filename = f"ode_guidance_{guidance_scale}.png"

        t = torch.linspace(0, 1, num_steps)[None, :].expand(batch_size, num_steps).to(device)
       
        x0 = torch.randn(batch_size, 1, image_size, image_size).to(device)
        # For each sample, pick a class (e.g., cycling through 0-9, 10 times each)
        labels = torch.arange(n_classes).repeat_interleave(samples_per_class)
        c = torch.zeros(batch_size, 1, n_classes, device=device)
        c[torch.arange(batch_size), 0, labels] = 1

        simulator = Euler(ode, t)
        xout = simulator.simulate(x0, c)
        grid = make_grid(xout, nrow=samples_per_class, normalize=True, value_range=(-1,1))
        save_image(grid, filename)
        # axes[idx].imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
        # axes[idx].axis("off")
        # axes[idx].set_title(f"Guidance: $w={guidance_scale:.1f}$", fontsize=25)

  