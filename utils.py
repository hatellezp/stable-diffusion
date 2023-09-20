import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import numpy as np

from typing import Tuple


def show_images(dataset: torch.utils.data.Dataset, num_samples: int = 20, cols: int = 4, show: bool = False) -> None:
    """ Plots some samples from the dataset """
    plt.figure(figsize=(15,15))
    for i, img in enumerate(dataset):
        if i == num_samples:
            break
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        plt.imshow(img[0])

    if show:
        plt.show()


def show_tensor_image(image: torch.Tensor, show: bool = False) -> None:
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))

    if show:
        plt.show()


def make_full_noise_sample(shape: Tuple[int, ...]) -> torch.Tensor:
    return torch.randn(shape)


def get_value_and_reshape(values: torch.Tensor, timestep: int | torch.Tensor, sample_shape: Tuple[int, ...]) -> torch.Tensor:
    if not (isinstance(timestep, int) or isinstance(timestep, torch.Tensor)):
        raise ValueError(f"Unexpected timestep type, expected int or torch.Tensor, got {type(timestep)}")

    if isinstance(timestep, int):
        timestep = torch.Tensor([timestep]).type(torch.int64)

    batch_size = timestep.shape[0]
    good_value = values.gather(-1, timestep)

    return good_value.reshape(batch_size, *((1,) * (len(sample_shape) - 1)))
























def sample_plot_image(IMG_SIZE, device, T):
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(img.detach().cpu())
    plt.show()


def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

