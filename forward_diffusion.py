import torch
from typing import Tuple

from utils import get_value_and_reshape


# The forward difussion processus that we use is derived from the DDPM
# paper: https://arxiv.org/pdf/2006.11239.pdf


def forward_diffusion(x0: torch.Tensor,
                      timestep: int,
                      alphas_bar_sqrt: torch.Tensor,
                      alphas_bar_one_minus: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    sample_shape = x0.shape

    # 1. Create gaussian noise (with standard distribution parameters)
    #    with the same shape as the sample.
    noise = torch.randn(sample_shape)  # torch.randn follows a standard normal distribution

    # 2. Get mean and variance, reshape on the fly.
    mean_for_timestep = get_value_and_reshape(alphas_bar_sqrt, timestep, sample_shape)
    variance_for_timestep = get_value_and_reshape(alphas_bar_one_minus, timestep, sample_shape)

    # 3. Diffuse the initial sample following the gaussian noise.
    xt = noise * variance_for_timestep + mean_for_timestep * x0

    return xt, noise
