import numpy as np
import torch

from typing import Dict


SCHEDULE_METHODS = (
    'linear',
    'cosine',
    'sqrt',
    'sqrt_linear',
    'log',
    'linear_cosine',
    'log_cosine',
    'clipped_cosine'
)

# Heavily inspired by runwayml/stable-diffusion. Most of methods are already mainstream,
# the relaxed cosine methods (`log_cosine` and `linear_cosine`) have a more slowly
# growing slope.
# `clipped_cosine` avoids to add almost no noise at the end by limiting the size of beta
# [DDPM](https://arxiv.org/pdf/2006.11239.pdf) set its range of betas to:
#   - beta_1 = 1e-4
#   - beta_T = 2e-2
# with T = 1000
def make_beta_schedule(schedule_type: str,
                       total_timesteps: int = 1000,
                       linear_start: float = 1e-4,
                       linear_end: float = 2e-2,
                       cosine_s: float = 4e-3,
                       cosine_clip: float = 4e-2,
                       device: str = None) -> torch.Tensor:

    if schedule_type not in SCHEDULE_METHODS:
        raise ValueError(f"schedule_type '{schedule_type}' unknown.")

    if schedule_type == "linear":
        betas = torch.linspace(linear_start, linear_end, total_timesteps, dtype=torch.float32)
    elif schedule_type == "cosine":
        timesteps = torch.arange(total_timesteps + 1, dtype=torch.float32) / total_timesteps + cosine_s

        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]

        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)
    elif schedule_type == 'clipped_cosine':
        betas = betas_cosine = make_beta_schedule('cosine', total_timesteps, linear_start, linear_end, cosine_s, to_numpy=False)
        betas = np.clip(betas, a_min=0, a_max=cosine_clip)
    elif schedule_type == "sqrt_linear":
        betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, total_timesteps, dtype=torch.float32) ** 2
    elif schedule_type == "sqrt":
        betas = torch.linspace(linear_start, linear_end, total_timesteps, dtype=torch.float32) ** 0.5
    elif schedule_type == 'log':
        betas = torch.log(torch.linspace(linear_start, linear_end, total_timesteps, dtype=torch.float32))

        betas_min = torch.abs(torch.min(betas))
        betas_range = torch.abs(torch.max(betas) - torch.min(betas))

        betas = (betas + betas_min) / betas_range
        betas = np.clip(betas, a_min=0, a_max=0.999)
    elif schedule_type == 'linear_cosine':
        betas_cosine = make_beta_schedule('cosine', total_timesteps, linear_start, linear_end, cosine_s, to_numpy=False)
        betas_linear = torch.linspace(betas_cosine[0], betas_cosine[-1], len(betas_cosine), dtype=torch.float32)

        betas = (betas_cosine + betas_linear) / 2
    elif schedule_type == 'log_cosine':
        betas_cosine = make_beta_schedule('cosine', total_timesteps, linear_start, linear_end, cosine_s, to_numpy=False)
        betas_log = torch.log(torch.linspace(linear_start, linear_end, total_timesteps, dtype=torch.float32))

        betas_min = torch.abs(torch.min(betas_log))
        betas_range = torch.abs(torch.max(betas_log) - torch.min(betas_log))

        betas_log = (betas_log + betas_min) / betas_range
        betas_log = np.clip(betas_log, a_min=0, a_max=0.999)

        betas = (betas_cosine + betas_log) / 2

    if device is not None:
        betas = betas.to(device)
    return betas


def make_alpha_from_beta(
        betas: torch.Tensor,
        device: str = None) -> Dict[str, torch.Tensor]:

    if isinstance(betas, torch.Tensor):
        alphas = 1. - betas
        alphas_sqrt_inverse = torch.sqrt(alphas)
        alphas_sqrt_inverse = 1. / alphas_sqrt_inverse
        alphas_bar = torch.cumprod(alphas, axis=0)
        alphas_bar_sqrt = torch.sqrt(alphas_bar)
        alphas_bar_sqrt_inverse = 1. / alphas_bar_sqrt
        alphas_bar_one_minus = 1. - alphas_bar
        alphas_bar_one_minus_sqrt = torch.sqrt(alphas_bar_one_minus)
        alphas_ratio = alphas_bar_one_minus / alphas_bar_one_minus_sqrt
        betas_ratio = betas / alphas_bar_one_minus_sqrt

        # define beta_bar here (from the DDPM paper https://arxiv.org/pdf/2006.11239.pdf)
        # I do not like this... find a better way to create the values
        betas_bar = [betas[0]]
        for idx in range(1, len(betas)):
            beta_bar = ((1 - alphas_bar[idx-1]) / (1 - alphas_bar[idx])) * betas[idx]
            betas_bar.append(beta_bar)

        betas_bar = torch.Tensor(betas_bar).type(torch.float32)

        betas_sqrt = torch.sqrt(betas)
        betas_bar_sqrt = torch.sqrt(betas_bar)
    else:
        raise ValueError(f"Incompatible `betas` type, expected torch.Tensor, got {type(betas)}")

    meanvar = {
        'alphas': alphas,
        'alphas_bar': alphas_bar,
        'alphas_bar_sqrt': alphas_bar_sqrt,
        'alphas_bar_sqrt_inverse': alphas_bar_sqrt_inverse,
        'alphas_bar_one_minus': alphas_bar_one_minus,
        'alphas_bar_one_minus_sqrt': alphas_bar_one_minus_sqrt,
        'alphas_ratio': alphas_ratio,
        'betas': betas,
        'betas_sqrt': betas_sqrt,
        'betas_bar': betas_bar,
        'betas_bar_sqrt': betas_bar_sqrt,
        'alphas_sqrt_inverse': alphas_sqrt_inverse,
        'betas_ratio': betas_ratio,
    }

    if device is not None:
        for parameter in meanvar:
            meanvar[parameter] = meanvar[parameter].to(device)

    return meanvar
