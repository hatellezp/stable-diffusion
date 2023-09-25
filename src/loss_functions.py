import torch
import torch.nn.functional as F

from torch import nn
from typing import Callable, Dict

from .forward_diffusion import ForwardDiffusion

class DiffusionLoss:
    def __init__(self,
                 loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 forward_diffusion: ForwardDiffusion):

        self.loss_function = loss_function
        self.forward_diffusion = forward_diffusion

    def __call__(self, model: nn.Module, x0: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        x_noisy, noise = self.forward_diffusion(x0, timestep)
        noise_pred = model(x_noisy, timestep)

        loss = self.loss_function(noise, noise_pred)
        return loss

    @property
    def meanvar_schedule(self) -> Dict[str, torch.Tensor]:
        return self.forward_diffusion.meanvar_schedule
