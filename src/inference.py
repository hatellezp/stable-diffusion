import matplotlib.pyplot as plt
import torch

from torch import nn
from typing import Dict

from .utils  import get_value_and_reshape, show_tensor_image, make_full_noise_sample


# This follows the algorithm 2 in https://arxiv.org/abs/2006.11239
def generate_sample_from_noise(
        model: nn.Module,
        xT: torch.Tensor,
        timestep: int,
        meanvar_schedule: Dict[str, torch.Tensor],
        variance_choice: int = 1,
        device: str = None) -> torch.Tensor:

    assert variance_choice in (1,2)
    shape = xT.shape

    # 2.1 Get some normal distributed noise.
    z = torch.randn(shape)
    if timestep <= 1:
        z = torch.zeros(shape)
    if device is not None:
        z = z.to(device)

    # 2.2 Read all computed mean and variance from the predefined schedule.
    alphas_sqrt_inverse_t = get_value_and_reshape(meanvar_schedule['alphas_sqrt_inverse'], timestep, shape, device=device)
    betas_ratio_t = get_value_and_reshape(meanvar_schedule['betas_ratio'], timestep, shape, device=device)

    if variance_choice == 1:
        variance_t = get_value_and_reshape(meanvar_schedule['betas_sqrt'], timestep, shape, device=device)
    else:
        variance_t = get_value_and_reshape(meanvar_schedule['betas_bar_sqrt'], timestep, shape, device=device)

    # 2.3 Predict noise at this timestep.
    predicted_noise = model(xT, timestep)
    if device is not None:
        predicted_noise = predicted_noise.to(device)

    # 2.4 xT receives the next denoised sample.
    xT = alphas_sqrt_inverse_t * (
        xT - betas_ratio_t * predicted_noise
    ) + variance_t * z

    return xT


def plot_image_sample(
        model: nn.Module,
        meanvar_schedule: Dict[str, torch.Tensor],
        img_size: int,
        img_channels: int,
        total_timesteps: int,
        num_images: int = 10,
        device: str = None,
        show_plot: bool = True) -> None:

    plt.figure(figsize=(15,15))
    plt.axis('off')
    stepsize = int(total_timesteps/num_images)

    shape = (1, img_channels, img_size, img_size)
    img = make_full_noise_sample(shape, device=device)

    for i in range(0,total_timesteps)[::-1]:
        if device == 'cuda':
            torch.cuda.empty_cache()

        timestep = torch.full((1,), i, device=device, dtype=torch.long)

        img = generate_sample_from_noise(model, img, timestep, meanvar_schedule, device=device)
        img = torch.clamp(img, -1.0, 1.0)

        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(img.detach().cpu(), show_plot=False)

    if show_plot:
        plt.show()
