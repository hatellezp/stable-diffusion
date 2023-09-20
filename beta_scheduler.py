import math
import numpy as np
import torch

from typing import Dict, Tuple


ArrayOrTensor = np.ndarray | torch.Tensor
SCHEDULE_METHODS = ('linear', 'cosine', 'sqrt', 'sqrt_linear', 'log', 'linear_cosine', 'log_cosine', 'clipped_cosine')

# Heavily inspired by runwayml/stable-diffusion. Most of methods are already mainstream,
# the relaxed cosine have a more slowly growing slope.
# [DDPM](https://arxiv.org/pdf/2006.11239.pdf) set its range of betas to:
#   - beta_1 = 1e-4
#   - beta_T = 2e-2
# with T = 1000
def make_beta_schedule(schedule_type: str,
                       timestep_nbr: int = 1000,
                       linear_start: float = 1e-4,
                       linear_end: float = 2e-2,
                       cosine_s: float = 4e-3,
                       cosine_clip: float = 4e-2,
                       to_numpy: bool = True) -> np.ndarray:

    if schedule_type not in SCHEDULE_METHODS:
        raise ValueError(f"schedule_type '{schedule_type}' unknown.")

    if schedule_type == "linear":
        betas = torch.linspace(linear_start, linear_end, timestep_nbr, dtype=torch.float64)
    elif schedule_type == "cosine":
        timesteps = torch.arange(timestep_nbr + 1, dtype=torch.float64) / timestep_nbr + cosine_s

        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]

        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)
    elif schedule_type == 'clipped_cosine':
        betas = betas_cosine = make_beta_schedule('cosine', timestep_nbr, linear_start, linear_end, cosine_s, to_numpy=False)
        betas = np.clip(betas, a_min=0, a_max=cosine_clip)
    elif schedule_type == "sqrt_linear":
        betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, timestep_nbr, dtype=torch.float64) ** 2
    elif schedule_type == "sqrt":
        betas = torch.linspace(linear_start, linear_end, timestep_nbr, dtype=torch.float64) ** 0.5
    elif schedule_type == 'log':
        betas = torch.log(torch.linspace(linear_start, linear_end, timestep_nbr, dtype=torch.float64))

        betas_min = torch.abs(torch.min(betas))
        betas_range = torch.abs(torch.max(betas) - torch.min(betas))

        betas = (betas + betas_min) / betas_range
        betas = np.clip(betas, a_min=0, a_max=0.999)
    elif schedule_type == 'linear_cosine':
        betas_cosine = make_beta_schedule('cosine', timestep_nbr, linear_start, linear_end, cosine_s, to_numpy=False)
        betas_linear = torch.linspace(betas_cosine[0], betas_cosine[-1], len(betas_cosine), dtype=torch.float64)

        betas = (betas_cosine + betas_linear) / 2
    elif schedule_type == 'log_cosine':
        betas_cosine = make_beta_schedule('cosine', timestep_nbr, linear_start, linear_end, cosine_s, to_numpy=False)
        betas_log = torch.log(torch.linspace(linear_start, linear_end, timestep_nbr, dtype=torch.float64))

        betas_min = torch.abs(torch.min(betas_log))
        betas_range = torch.abs(torch.max(betas_log) - torch.min(betas_log))

        betas_log = (betas_log + betas_min) / betas_range
        betas_log = np.clip(betas_log, a_min=0, a_max=0.999)

        betas = (betas_cosine + betas_log) / 2

    return betas.numpy() if to_numpy else betas


def make_alpha_from_beta(betas: ArrayOrTensor, to_numpy: bool = True, device: str = 'cpu') -> Dict[str, ArrayOrTensor]:
    if isinstance(betas, np.ndarray) or isinstance(betas, torch.Tensor):
        module = np if isinstance(betas, np.ndarray) else torch

        alphas = 1. - betas
        alphas_bar = module.cumprod(alphas, axis=0)
        alphas_bar_sqrt = module.sqrt(alphas_bar)
        alphas_bar_one_minus = 1. - alphas_bar

        # define beta_bar here (from the DDPM paper https://arxiv.org/pdf/2006.11239.pdf)

        # I do not like this... find a better way to create the values
        betas_bar = [betas[0]]
        for idx in range(1, len(betas)):
            beta_bar = ((1 - alphas_bar[idx-1]) / (1 - alphas_bar[idx])) * betas[idx]
            betas_bar.append(beta_bar)

        if isinstance(betas, np.ndarray):
            betas_bar = np.asarray(beta_bar, dtype=np.float64)  # is this the good type ?
        else:
            betas_bar = torch.Tensor(betas_bar).type(torch.float64)

        betas_sqrt = module.sqrt(betas)
        betas_bar_sqrt = module.sqrt(betas_bar)
    else:
        raise ValueError(f"Incompatible `betas` type, expected numpy.ndarray or torch.Tensor, got {type(betas)}")

    if to_numpy and isinstance(betas, torch.Tensor):
        alphas = alphas.numpy()
        alphas_bar = alphas_bar.numpy()
        alphas_bar_sqrt = alphas_bar_sqrt.numpy()
        alphas_bar_one_minus = alphas_bar_one_minus.numpy()
        betas_bar = betas_bar.numpy()
        betas_bar_sqrt = betas_bar_sqrt.numpy()

    hyperparameters = {
        'alphas': alphas,
        'alphas_bar': alphas_bar,
        'alphas_bar_sqrt': alphas_bar_sqrt,
        'alphas_bar_one_minus': alphas_bar_one_minus,
        'betas': betas,
        'betas_sqrt': betas_sqrt,
        'betas_bar': betas_bar,
        'betas_bar_sqrt': betas_bar_sqrt,
    }

    if not to_numpy:
        for parameter in hyperparameters:
            hyperparameters[parameter].to(device)

    return hyperparameters
