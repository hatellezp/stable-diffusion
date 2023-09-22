import matplotlib.pyplot as plt
import torch

from torch import nn
from tqdm import tqdm
from typing import Dict

from .tensor_types import ArrayOrTensor
from .utils  import get_value_and_reshape, show_tensor_image


# This follows the algorithm 2 in https://arxiv.org/abs/2006.11239
def generate_sample_from_noise(
        model: nn.Module,
        xT: torch.Tensor,
        total_timesteps: int,
        hyperparameters_schedule: Dict[str, ArrayOrTensor],
        variance_choice: int = 1,
        device: str = None) -> torch.Tensor:

    assert variance_choice in (1,2)

    shape = xT.shape

    # 1. Set the model to eval, no gradient evaluation.
    model.eval()

    # 2. Accumulate the denoised sample on xT.
    for i in tqdm(range(total_timesteps)):
        timestep = total_timesteps - i

        # 2.1 Get some normal distributed noise.
        # This seems weird to me ....
        z = torch.randn(shape)
        if timestep <= 1:
            z = torch.zeros(shape)
        if device is not None:
            z = z.to(device)

        # 2.2 Read all computed mean and variance from the predefined schedule.
        alphas_bar_sqrt_inverse_t = get_value_and_reshape(hyperparameters_schedule['alphas_bar_sqrt_inverse'], timestep, shape, device=device)
        alphas_ratio_t = get_value_and_reshape(hyperparameters_schedule['alphas_ratio'], timestep, shape, device=device)

        if variance_choice == 1:
            variance_t = get_value_and_reshape(hyperparameters_schedule['betas_sqrt'], timestep, shape, device=device)
        else:
            variance_t = get_value_and_reshape(hyperparameters_schedule['betas_bar_sqrt'], timestep, shape, device=device)

        # 2.3 Predict noise at this timestep.
        predicted_noise = model(xT, timestep)
        if device is not None:
            predicted_noise = predicted_noise.to(device)

        # 2.4 xT receives the next denoised sample.
        xT = alphas_bar_sqrt_inverse_t * (
            xT - alphas_ratio_t * predicted_noise
        ) + variance_t * z

    return xT


def plot_image_sample(
        model: nn.Module,
        hyperparameters_schedule: Dict[str, ArrayOrTensor],
        img_size: int,
        total_timesteps: int,
        num_images: int = 10,
        device: str = None,
        show_plot: bool = True) -> None:

    plt.figure(figsize=(15,15))
    plt.axis('off')
    stepsize = int(total_timesteps/num_images)

    img = torch.randn((1, 3, img_size, img_size), device=device)

    for i in range(0,total_timesteps)[::-1]:
        timestep = torch.full((1,), i, device=device, dtype=torch.long)
        img = generate_sample_from_noise(model, img, timestep, hyperparameters_schedule, device=device)

        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)

        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(img.detach().cpu(), show_plot=False)

    if show_plot:
        plt.show()
