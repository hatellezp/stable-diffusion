import torch

from torch import nn
from typing import Dict, Tuple

from utils import ArrayOrTensor, get_value_and_reshape

# This follows the algorithm 2 in https://arxiv.org/abs/2006.11239
def algorithm2_ddpm(
        model: nn.Module,
        xT: torch.Tensor,
        shape: Tuple[int,...],
        total_timesteps: int,
        hyperparameters_schedule: Dict[str, ArrayOrTensor],
        variance_choise: int = 1) -> torch.Tensor:

    # 1. Set the model to eval, no gradient evaluation.
    model.eval()

    # 2. Accumulate the denoised image on xT.
    for i in range(total_timesteps):
        timestep = total_timesteps - i

        # 2.1 Get some normal distributed noise.
        # This seems weird to me ....
        z = torch.randn(shape)
        if timestep <= 1:
            z = torch.zeros(shape)

        # 2.2 Read all computed mean and variance from the predefined schedule.
        alpha_bar_sqrt_inverse_t = get_value_and_reshape(hyperparameters_schedule['alpha_bar_sqrt_inverse'], timestep, shape)
        alpha_ratio_t = get_value_and_reshape(hyperparameters_schedule['alphas_ration'], timestep, shape)

        if variance_choise == 1:
            variance_t = get_value_and_reshape(hyperparameters_schedule['betas_sqrt'], timestep, shape)
        else:
            variance_t = get_value_and_reshape(hyperparameters_schedule['betas_bar_sqrt'], timestep, shape)

        # 2.3 Predict noise at this timestep.
        predicted_noise = model(xT, timestep)

        # 2.4 xT received the next denoised image.
        xT = alpha_bar_sqrt_inverse_t * (
            xT - alpha_ratio_t * predicted_noise
        ) + variance_t * z

    return xT
