"""
Load a trained BEAM flow model once and turn conditioning vectors into images.

Draft: reuses the research code in beam/ (the repo root must be importable).
"""

import glob
import os
import pickle
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from beam.models.unet import ContextUnet  # noqa: E402
from beam.models.interpolant import ScoreMatch, EMA  # noqa: E402
from beam.models.probabilitypath import GaussianProbabilityPath, OTAlpha, OTBeta  # noqa: E402
from beam.utils.config import load_config, flatten_config  # noqa: E402
from torchdiffeq import odeint  # noqa: E402

# training images are (FFI - quiet background) / 633118 in 30-min-cadence units
# (EXPTIME 1425.6 s); this turns de-normalized model output into e-/s per pixel
E_PER_S_PER_UNIT = 633118 / 1425.6

DEFAULT_MODEL_DIR = os.environ.get(
    "TESSBEAM_MODEL_DIR",
    "/pdo/users/djtufto/BEAM/model_outputs/cam3_tica_star_256x256_v2",
)


SOLVER = "bosh3"
TOLERANCE = 1e-2


def _worst_pixel(err):
    """The solver's step-size control: the largest error of any pixel in any frame of the batch.
    torchdiffeq's default is one RMS over the whole batch"""
    return err.abs().max()


class _ConditionalFlow:
    """Velocity field = conditional network output.

    Classifier-free guidance at scale 1 is (1-1)*uncond + 1*cond = cond, so the
    unconditional half of the batch can be skipped exactly.
    """

    def __init__(self, nn_model, cam):
        self.nn_model = nn_model
        self.cam = cam

    def velocity_field(self, x, c, t):
        t = t.clamp(1e-3, 1.0 - 1e-3)
        return self.nn_model(x, c, t, torch.zeros_like(c), self.cam)


def _find_checkpoint(model_dir):
    finals = sorted(glob.glob(os.path.join(model_dir, "checkpoint_epoch_final_*.pth")))
    if finals:
        return finals[-1]
    ckpts = glob.glob(os.path.join(model_dir, "checkpoint_*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint_*.pth in {model_dir}")
    return max(ckpts, key=os.path.getmtime)


class StrayLightModel:
    def __init__(self, model_dir=DEFAULT_MODEL_DIR, device=None, checkpoint=None):
        self.model_dir = model_dir
        self.device = torch.device(
            device or os.environ.get("TESSBEAM_DEVICE")
            or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        cfg = flatten_config(load_config(os.path.join(model_dir, "config.yaml")))
        if cfg["model_architecture"] != "flow":
            raise NotImplementedError("the draft only samples flow models")
        with open(os.path.join(model_dir, "context_stats.pkl"), "rb") as f:
            self.context_stats = pickle.load(f)
        with open(os.path.join(model_dir, "image_stats.pkl"), "rb") as f:
            img = pickle.load(f)
        self.img_mean, self.img_std = float(img["mean"]), float(img["std"])
        self.image_shape = tuple(cfg["data_image_shape"])
        self.epsilon = float(cfg.get("model_epsilon", 1e-4))
        self.trained_cameras = {int(cfg.get("data_camera_number", 3))}
        self.checkpoint = checkpoint or _find_checkpoint(model_dir)

        unet = ContextUnet(
            in_feats=1,
            context_len=len(self.context_stats["mean"]),
            n_feat=cfg["model_n_feat"],
            channel_mults=cfg["model_channel_mults"],
            heads_at=cfg["model_heads_at"],
            time_dim=cfg.get("model_time_dim", 128),
            context_dim=cfg.get("model_context_dim", 64),
            emb_dim=cfg.get("model_emb_dim", 256),
            num_res=cfg["model_num_res"],
            use_cross_attn=cfg.get("model_use_cross_attn", True),
        )
        model = ScoreMatch(
            nn_model=unet,
            probability_path=GaussianProbabilityPath(alpha=OTAlpha(), beta=OTBeta()),
            device=self.device,
            drop_prob=cfg.get("model_drop_prob", 0.1),
            epsilon=self.epsilon,
            architecture="flow",
        )
        state = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        ema = EMA(model=model)                     # sample with the EMA weights
        ema.load_state_dict(state["ema_state_dict"])
        ema.to(self.device)
        ema.copy_to(model)
        model.eval()
        self.nn_model = model.nn_model

    @torch.no_grad()
    def sample(self, ctx, camera=3, seed=0, batch_size=16, progress=None, solver=SOLVER, tolerance=TOLERANCE):
        """
        ctx: (N, 16) standardized features (tessbeam.features.context_vectors).
        Returns (N, H, W) float32 images in e-/s per pixel.
        """
        if camera not in self.trained_cameras:
            raise ValueError(f"camera {camera} not trained; this model covers {sorted(self.trained_cameras)}")
        ctx = np.asarray(ctx, dtype=np.float32).reshape(-1, len(self.context_stats["mean"]))
        n = len(ctx)
        gen = torch.Generator().manual_seed(int(seed))
        x0 = torch.randn(1, 1, *self.image_shape, generator=gen)
        # Reduce variance for point estimate. Always start at 0 mean noise
        x0 = (x0 - x0.mean()).to(self.device)
        t = torch.tensor([self.epsilon, 1.0 - self.epsilon], device=self.device)
        out = np.empty((n, *self.image_shape), dtype=np.float32)
        for s in range(0, n, batch_size):
            c = torch.from_numpy(ctx[s:s + batch_size]).to(self.device).unsqueeze(1)   # (B, 1, 16)
            b = c.shape[0]
            cam = torch.full((b,), camera - 1, dtype=torch.long, device=self.device)
            flow = _ConditionalFlow(self.nn_model, cam)
            # step_t: land exactly on the end time. 
            xs = odeint(lambda tt, x: flow.velocity_field(x, c, tt.expand(b, 1, 1, 1)),
                        x0.expand(b, -1, -1, -1).clone(), t, method=solver, atol=tolerance, rtol=tolerance,
                        options=dict(norm=_worst_pixel, step_t=[1.0 - self.epsilon]))
            x = xs[-1][:, 0].float().cpu().numpy()
            out[s:s + b] = (x * self.img_std + self.img_mean) * E_PER_S_PER_UNIT
            if progress is not None:
                progress(min(s + b, n), n)
        return out
