import torch

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from .utils import get_value_and_reshape


# The forward difussion process that we use is derived from the DDPM
# paper: https://arxiv.org/pdf/2006.11239.pdf


class ForwardDiffusion(ABC):
    def __init__(self, meanvar_schedule: Dict[str, torch.Tensor], device: Optional[str] = None):
        self._meanvar_schedule = meanvar_schedule
        self.device = device

    @abstractmethod
    def __call__(self, x0: torch.Tensor, timestep: torch.Tensor) -> Tuple[torch.Tensor,...]:
        pass

    @property
    def meanvar_schedule(self) -> Dict[str, torch.Tensor]:
        return self._meanvar_schedule

    @meanvar_schedule.setter
    def meanvar_schedule(self, value):
        self._meanvar_schedule = value

class ForwardDiffusionDDPM(ForwardDiffusion):
    def __call__(self, x0: torch.Tensor, timestep: torch.Tensor) -> Tuple[torch.Tensor, ...]:

        sample_shape = x0.shape
        alphas_bar_sqrt = self.meanvar_schedule['alphas_bar_sqrt']
        alphas_bar_one_minus_sqrt = self.meanvar_schedule['alphas_bar_one_minus_sqrt']

        # 1. Create gaussian noise (with standard distribution parameters)
        #    with the same shape as the sample.
        noise = torch.randn(sample_shape)  # torch.randn follows a standard normal distribution
        if self.device is not None:
            noise = noise.to(self.device)

        # 2. Get mean and variance, reshape on the fly.
        mean_for_timestep = get_value_and_reshape(alphas_bar_sqrt, timestep, sample_shape, device=self.device)
        variance_for_timestep = get_value_and_reshape(alphas_bar_one_minus_sqrt, timestep, sample_shape, device=self.device)

        # 3. Diffuse the initial sample following the gaussian noise.
        xt = noise * variance_for_timestep + mean_for_timestep * x0

        return xt, noise
