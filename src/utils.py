import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image
from torchvision import transforms
from typing import Tuple

ArrayOrTensor = np.ndarray | torch.Tensor


def image_from_path_to_tensor(
        path: str,
        resize_h: int,
        resize_w: int,
        random_flip: bool = True) -> torch.Tensor:

    # 1. Define transformation for the image.
    data_transforms = [transforms.Resize((resize_h, resize_w))]
    if random_flip:
        data_transforms.append(transforms.RandomHorizontalFlip())
    data_transforms.extend([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    data_transform = transforms.Compose(data_transforms)

    # 2. Open, convert and return.
    image = Image.open(path).convert("RGB")
    return data_transform(image)


def show_images(dataset: torch.utils.data.Dataset,
                num_samples: int = 20,
                cols: int = 4,
                show_plot: bool = True,
                figsize: Tuple[int, int] = (20, 20)) -> None:
    """ Plots some samples from the dataset """

    plt.figure(figsize=figsize)
    for i, img in enumerate(dataset):
        if i == num_samples:
            break
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        plt.imshow(img[0])

    if show_plot:
        plt.show()


def default_tensor_to_image_transform():
    return transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])


def show_tensor_image(
        image: torch.Tensor,
        reverse_transforms: transforms.Compose = None,
        show_plot: bool = True,
        ax=None) -> None:

    if reverse_transforms is None:
        reverse_transforms = default_tensor_to_image_transform()

    # If the image is a batch of images then only take the first.
    if len(image.shape) == 4:
        image = image[0, :, :, :]

    # Plot following a specific ax if provided.
    if ax is not None:
        ax.imshow(reverse_transforms(image))
    else:
        plt.imshow(reverse_transforms(image))

    if show_plot:
        plt.show()


def make_full_noise_sample(
        shape: Tuple[int, ...],
        manual_seed: int = None,
        iterate: int = None) -> torch.Tensor:

    # You can guaranteed reproducibilty of your experiment by forcing a manual
    # seed for the random generator in torch. For a little flexibility in this
    # process you can ask it to generate `iterate - 1` samples and discard them
    # before returning.

    if manual_seed:
        torch.manual_seed(manual_seed)

    if iterate and manual_seed:
        for _ in range(iterate - 1): torch.randn(shape)

    return torch.randn(shape)


def get_value_and_reshape(
        values: torch.Tensor,
        timestep: ArrayOrTensor,
        sample_shape: Tuple[int, ...]) -> torch.Tensor:

    if not (isinstance(timestep, int) or isinstance(timestep, torch.Tensor)):
        raise ValueError(
            f"Unexpected timestep type, expected int or torch.Tensor, got {type(timestep)}"
        )

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


