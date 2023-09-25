import math
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch

from torch import nn
from torch import optim
from typing import  Any, Dict, Optional, Tuple
from pathlib import Path

from .loss_functions import DiffusionLoss
from .utils import get_value_and_reshape

# The statement below is not entirely true...
# Following the structure from 'U-Net for Brain MRI' in pytorch
# https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/

# I'm under the impression that this unet is thought for imagerie,
# I will need to adapt it to be data agnostic.
class UNetBlock(pl.LightningModule):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 time_emb_dim: int,
                 use_gn: bool = False,
                 gn_size: int = 32,
                 up: bool = False,) -> None:
        super().__init__()

        # 1. Time embedding (find a better way of doing this).
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        # 2. If block is going up, adapt.
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 3, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 3, padding=1)


        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # Group Normalization adds a considerable speed up to training, see https://arxiv.org/abs/1803.08494
        # While near enough Batch Normalization, it suffer from a litte disavantage in generalization.
        if use_gn:
            self.norm1 = nn.GroupNorm(gn_size, out_ch)
            self.norm2 = nn.GroupNorm(gn_size, out_ch)
        else:
            self.norm1 = nn.BatchNorm2d(out_ch)
            self.norm2 = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # 1. First Conv
        h = self.norm1(self.relu(self.conv1(x)))

        # 2. Time embedding, and extend last 2 dimensions
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]

        # 3. Add time channel
        h = h + time_emb

        # 4. Second Conv
        h = self.norm2(self.relu(self.conv2(h)))

        # 5. Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(pl.LightningModule):
    def __init__(self, time_emb_dim: int, time_emb_limit: int = 10000, device: Optional[str] = None):
        super().__init__()

        self.dim = time_emb_dim
        self.limit = time_emb_limit

        self.half_dim = self.dim // 2
        self.embeddings = math.log(self.limit) / (self.half_dim - 1)
        self.negative_embeddings = - self.embeddings
        self.embeddings = torch.exp(
            torch.arange(self.half_dim) * self.negative_embeddings
        )

        if device is not None:
            self.embeddings = self.embeddings.to(device)

    def forward(self, time: torch.Tensor):
        embeddings = self.embeddings.to(time.device)
        embeddings = time[:, None] * self.embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings

class SimpleUnet(pl.LightningModule):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self,
                 image_channels: int = 3,
                 schedule: Tuple[int, ...] = (64, 128, 256, 512, 1024),
                 out_dim: int = 3,
                 time_emb_dim: int = 32,
                 time_emb_limit: int = 10000,
                 use_gn: Optional[Tuple[bool,...]] = None,
                 gn_size: Optional[Tuple[int,...]] = None,
                 device: Optional[str] = None):
        super().__init__()

        self.image_channels = image_channels
        self.down_channels = schedule
        self.up_channels = schedule[::-1]
        self.out_dim = out_dim
        self.time_emb_dim = time_emb_dim
        self.trajectory_length = len(schedule)

        if use_gn is not None and len(use_gn) != len(schedule) - 1:
            raise ValueError(f"Mismatched leng for schedule and use of group normalization.")
        if use_gn == None:
            use_gn = (False,) * (len(schedule) - 1)
        self.use_gn = use_gn

        if gn_size is not None and len(gn_size) != len(schedule) - 1:
            raise ValueError(f"Mismatched length for schedule and use of group normalization size.")
        if gn_size == None:
            gn_size = (32,) * (len(schedule) - 1)
        self.gn_size = gn_size

        # 1. Define the time embedding layer.
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(
                    time_emb_dim=time_emb_dim, time_emb_limit=time_emb_limit, device=device),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )

        # 2. The initial transformation to enter in the unet is not a
        #    factory unet block
        self.input_layer = nn.Conv2d(image_channels, self.down_channels[0], 3, padding=1)

        # 3. Down trajectory.
        self.downs = nn.ModuleList([
            UNetBlock(
                in_ch=self.down_channels[i],
                out_ch=self.down_channels[i+1],
                time_emb_dim=self.time_emb_dim,
                use_gn=self.use_gn[i],
                gn_size=self.gn_size[i],
            ) for i in range(self.trajectory_length - 1)
        ])

        # 4. Up trajectory.
        self.ups = nn.ModuleList([
            UNetBlock(
                self.up_channels[i],
                self.up_channels[i+1],
                time_emb_dim,
                use_gn=self.use_gn[i],
                gn_size=self.gn_size[i],
                up=True)
            for i in range(self.trajectory_length - 1)
        ])

        # 5. Output layer.
        self.output_layer = nn.Conv2d(self.up_channels[-1], self.image_channels, 1)

    def forward(self, x, timestep):

        if not isinstance(timestep, torch.Tensor):
            timestep = torch.Tensor([timestep]).type(torch.int64)
        timestep = timestep.to(self.device)

        # 1. Embedd time
        t = self.time_mlp(timestep)

        # 2. Prepare the sample to enter the unet
        x = self.input_layer(x)

        # TODO: find a way to compute the shape of residuals, it would like to avoid used an ugly list for this.
        # 3. Unet going down, residual inputs to connect to the corresponding up block.
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)  # Time embedding is added to each block
            residual_inputs.append(x)

        # 4. Unet going up, add the residual inputs.
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)

        # 5. Return the output layer result.
        return self.output_layer(x)


class DDPM(pl.LightningModule):

    def __init__(self,
            diffusion_model: nn.Module,
            loss_function: DiffusionLoss,
            total_timesteps: int = 100,
    ):
        super().__init__()

        self.total_timesteps = total_timesteps
        self.diffusion_model = diffusion_model
        self.meanvar_schedule = loss_function.meanvar_schedule

        self.loss_function = lambda x0, timestep: loss_function(self.diffusion_model, x0, timestep)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x0 = batch[0]
        device = x0.device

        timestep = torch.randint(0, self.total_timesteps, (len(x0), )).long().to(device)
        loss = self.loss_function(x0, timestep)

        self.log("train loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x0 = batch[0]
        device = x0.device

        timestep = torch.randint(0, self.total_timesteps, (len(x0), )).long().to(device)
        loss = self.loss_function(x0, timestep)

        self.log("val_loss", loss)

    def full_denoise(self, xT: torch.Tensor, variance_choice: int = 1, step_size: Optional[int] = None, device: Optional[str] = None) -> torch.Tensor:

        assert variance_choice in (1,2)
        shape = xT.shape
        total_timesteps = len(self.meanvar_schedule['betas'])

        step_size = 1 if step_size is None else step_size

        for timestep in range(0, total_timesteps, step_size):
            timestep = total_timesteps - timestep - 1

            # 2.1 Get some normal distributed noise.
            z = torch.randn(shape)
            if timestep <= 1:
                z = torch.zeros(shape)
            if device is not None:
                z = z.to(device)
            timestep = torch.Tensor([timestep]).type(torch.int64).to(xT.device)

            # 2.2 Read all computed mean and variance from the predefined schedule.
            alphas_sqrt_inverse_t = get_value_and_reshape(self.meanvar_schedule['alphas_sqrt_inverse'], timestep, shape, device=device)
            betas_ratio_t = get_value_and_reshape(self.meanvar_schedule['betas_ratio'], timestep, shape, device=device)

            if variance_choice == 1:
                variance_t = get_value_and_reshape(self.meanvar_schedule['betas_sqrt'], timestep, shape, device=device)
            else:
                variance_t = get_value_and_reshape(self.meanvar_schedule['betas_bar_sqrt'], timestep, shape, device=device)

            # 2.3 Predict noise at this timestep.
            predicted_noise = self.diffusion_model(xT, timestep)
            if device is not None:
                predicted_noise = predicted_noise.to(device)

            # 2.4 xT receives the next denoised sample.
            xT = alphas_sqrt_inverse_t * (
                xT - betas_ratio_t * predicted_noise
            ) + variance_t * z

        return xT

    def save_state_dict(self, parameters_path: str | Path) -> None:
        torch.save(self.state_dict(), parameters_path)
