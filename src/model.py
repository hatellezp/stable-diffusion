import math
import torch

from torch import nn
from typing import Tuple
from pathlib import Path


# The statement below is not entirely true...
# Following the structure from 'U-Net for Brain MRI' in pytorch
# https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/

# I'm under the impression that this unet is thought for imagerie,
# I will need to adapt it to be data agnostic.
class UNetBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 time_emb_dim: int,
                 use_gn: bool = False,
                 gn_size: int = 32,
                 use_pooling: bool = False,
                 up: bool = False,) -> None:
        super().__init__()

        # 1. Time embedding (find a better way of doing this).
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        # 2. If block is going up, adapt.
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)


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

    def forward(self, x, t, ):
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


# TODO: see how this works and try to explain it to others !
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int, limit: int = 10000):
        super().__init__()

        self.dim = dim
        self.limit = limit

    def forward(self, time: torch.Tensor):
        # 1. Prepare current device and set half dimension
        device = time.device
        half_dim = self.dim // 2

        embeddings = math.log(self.limit) / (half_dim - 1)
        negative_embeddings = - embeddings
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * negative_embeddings
            )
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        # TODO: Double check the ordering here
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self,
                 image_channels: int = 3,
                 schedule: Tuple[int, ...] = (64, 128, 256, 512, 1024),
                 out_dim: int = 3,
                 time_emb_dim: int = 32,
                 device: str = None):
        super().__init__()

        self.image_channels = image_channels
        self.down_channels = schedule
        self.up_channels = schedule[::-1]
        self.out_dim = out_dim
        self.time_emb_dim = time_emb_dim
        self.trajectory_length = len(schedule)
        self.device = device

        # 1. Define the time embedding layer.
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )

        # 2. The initial transformation to enter in the unet is not a
        #    factory unet block
        self.input_layer = nn.Conv2d(image_channels, self.down_channels[0], 3, padding=1)

        # 3. Down trajectory.
        self.downs = nn.ModuleList([
            UNetBlock(self.down_channels[i], self.down_channels[i+1], self.time_emb_dim)
            for i in range(self.trajectory_length - 1)
        ])

        # 4. Up trajectory.
        self.ups = nn.ModuleList([
            UNetBlock(self.up_channels[i], self.up_channels[i+1], time_emb_dim, up=True)
            for i in range(self.trajectory_length - 1)
        ])

        # 5. Output layer.
        # self.output_layer = nn.Conv2d(self.up_channels[-1], out_dim, 1)
        self.output_layer = nn.Conv2d(self.up_channels[-1], self.image_channels, 1)



    def forward(self, x, timestep):

        if not isinstance(timestep, torch.Tensor):
            timestep = torch.Tensor([timestep]).type(torch.int64)
        if self.device is not None:
            timestep = timestep.to(self.device)

        # 1. Embedd time
        t = self.time_mlp(timestep)

        # 2. Prepare the sample to enter the unet
        x = self.input_layer(x)

        # 3. Unet going down, residual inputs to connect to the corresponding
        #    up block.
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

    @staticmethod
    def from_pretrained(
        parameters_path: str | Path,
        image_channels: int = 3,
        schedule: Tuple[int, ...] = (64, 128, 256, 512, 1024),
        out_dim: int = 3,
        time_emb_dim: int = 32,
        device: str = None) -> 'SimpleUnet':

        model = SimpleUnet(image_channels=image_channels, schedule=schedule, out_dim=out_dim, time_emb_dim=time_emb_dim, device=device)
        model.load_state_dict(torch.load(parameters_path))

        return model

    def save_state_dict(self, parameters_path: str | Path) -> None:
        torch.save(self.state_dict(), parameters_path)
