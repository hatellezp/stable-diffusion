import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from PIL import Image
from torch import nn
from torchvision import transforms
from typing import Dict, Tuple, Optional


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


def default_image_to_tensor_transform(
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

    return data_transform


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
        reverse_transform: transforms.Compose = None,
        show_plot: bool = True,
        ax=None) -> None:

    if reverse_transform is None:
        reverse_transform = default_tensor_to_image_transform()

    # If the image is a batch of images then only take the first.
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    image = reverse_transform(image)

    # Plot following a specific ax if provided.
    if ax is not None:
        ax.imshow(image)
    else:
        plt.imshow(image)

    if show_plot:
        plt.show()


def make_full_noise_sample(
        shape: Tuple[int, ...],
        manual_seed: int = None,
        iterate: int = None,
        device: str = None) -> torch.Tensor:

    # You can guaranteed reproducibilty of your experiment by forcing a manual
    # seed for the random generator in torch. For a little flexibility in this
    # process you can ask it to generate `iterate - 1` samples and discard them
    # before returning.

    if manual_seed:
        torch.manual_seed(manual_seed)

    if iterate and manual_seed:
        for _ in range(iterate - 1): torch.randn(shape)

    return torch.randn(shape, device=device, dtype=torch.float32)


def get_value_and_reshape(
        values: torch.Tensor,
        timestep: torch.Tensor,
        sample_shape: Tuple[int, ...],
        device: str = None) -> torch.Tensor:

    if device is not None:
        values = values.to(device)
        timestep = timestep.to(device)

    batch_size = timestep.shape[0]
    good_value = values.gather(-1, timestep)

    return good_value.reshape(batch_size, *((1,) * (len(sample_shape) - 1)))


def gather_data(dataset: str,
                dataset_path: str,
                download: bool = True,
                separate_test_train: bool = True,
                transform: Optional[transforms.Compose] = None) -> torch.utils.data.Dataset:

    dataset_module = {
        'oxford': torchvision.datasets.OxfordIIITPet,
        'cifar10': torchvision.datasets.CIFAR10,
        'cifar100': torchvision.datasets.CIFAR100,
        'mnist': torchvision.datasets.MNIST,
        'fashionmnist': torchvision.datasets.FashionMNIST,
        'flowers': torchvision.datasets.Flowers102,
    }

    if dataset not in dataset_module:
        raise ValueError(f"Unexpected dataset: {dataset}")

    module_to_load = dataset_module[dataset]
    if separate_test_train:
        data = (
            module_to_load(root=dataset_path, train=True, download=True, transform=transform),
            module_to_load(root=dataset_path, train=False, download=True, transform=transform)
        )
    else:
        data = module_to_load(root=dataset_path, download=True, transform=transform)

    return data
