import matplotlib.pyplot as plt
import torch
import torchvision

from tqdm import tqdm
from typing import List, Tuple

from .beta_scheduler import SCHEDULE_METHODS, make_alpha_from_beta, make_beta_schedule
from .forward_diffusion import forward_diffusion_ddpm
from .utils import image_from_path_to_tensor, show_tensor_image, make_full_noise_sample, show_images


def visualize_noise_adding_methods(
        image_path='ressources/images/beatiful_fox.jpeg',
        num_images=20, T=500,
        show_plot=True) -> None:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    image = image_from_path_to_tensor(image_path, 128, 128, random_flip=True)
    image.to(device)

    stepsize = int(T/num_images)
    methods_length = len(SCHEDULE_METHODS)

    plt.figure(figsize=(20,20))
    plt.axis('off')
    _, axs = plt.subplots(methods_length, num_images)


    for index in range(methods_length):
        method = SCHEDULE_METHODS[index]
        print('=================================')
        print(f'Method: {method}')

        betas = make_beta_schedule(method, total_timesteps=T, to_numpy=False)
        meanvar = make_alpha_from_beta(betas, to_numpy=False, device=device)
        print("meanvar created")

        for idx in tqdm(range(0, num_images), unit='img'):
            timestep = idx*stepsize
            img, _ = forward_diffusion_ddpm(
                image, timestep, meanvar['alphas_bar_sqrt'], meanvar['alphas_bar_one_minus']
            )
            show_tensor_image(img, ax=axs[index, idx], show_plot=False)

        axs[index, 0].set_title(method)

    plt.legend()
    if show_plot:
        plt.show()


def visualize_full_noise(image_h: int = 128, image_w: int = 128):
    noise = make_full_noise_sample((3, image_h, image_w))
    show_tensor_image(noise)


def visualize_dataset_sample(
          dataset: str = 'mnist',
          dataset_path: str = "../../../Datasets",
          download: bool = True,
          num_samples: int = 20,
          cols: int = 4,
          show_plot: bool = True,
          figsize: Tuple[int, int] = (20, 20)) -> None:

    if dataset == 'oxford':
        data = torchvision.datasets.OxfordIIITPet(root=dataset_path, download=download)
    elif dataset == 'cifar10':
        data = torchvision.datasets.CIFAR10(root=dataset_path, download=download)
    elif dataset == 'cifar100':
        data = torchvision.datasets.CIFAR100(root=dataset_path, download=download)
    elif dataset == 'mnist':
        data = torchvision.datasets.MNIST(root=dataset_path, download=download)
    elif dataset == 'fashionmnist':
        data = torchvision.datasets.FashionMNIST(root=dataset_path, download=download)
    elif dataset == 'flowers':
        data = torchvision.datasets.Flowers102(root=dataset_path, download=download)
    else:
        raise ValueError(f"Unexpected dataset: {dataset}")
    show_images(data, num_samples=num_samples, cols=cols, show_plot=show_plot, figsize=figsize)


def visualize_noise_mean_variance_schedules(timesteps: List[int] = None, show_plot: bool = True):
    schedules = SCHEDULE_METHODS
    if timesteps is None:
        timesteps = [100, 1000, 10000]

    schedules_length = len(schedules)
    timesteps_length = len(timesteps)

    _, axs = plt.subplots(timesteps_length, 5)


    for idx in range(timesteps_length):

        T = timesteps[idx]
        l = list(range(1, T + 1))

        for schedule in schedules:
            betas = make_beta_schedule(schedule, total_timesteps=T)
            meanvar = make_alpha_from_beta(betas)

            axs[idx, 0].plot(l, betas, label=schedule)
            axs[idx, 1].plot(l, meanvar['alphas'], label=schedule)
            axs[idx, 2].plot(l, meanvar['alphas_bar'], label=schedule)
            axs[idx, 3].plot(l, meanvar['alphas_bar_sqrt'], label=schedule)
            axs[idx, 4].plot(l, meanvar['alphas_bar_one_minus'], label=schedule)



        if idx > 0:
            axs[idx, 0].set_title(f"A time schedule with T={T}")

    axs[0, 0].set_title(f"A time schedule with T={timesteps[0]}.\nbetas, smaller means more noise.")
    axs[0, 1].set_title(f"1 - betas")
    axs[0, 2].set_title(f"alphas bar")
    axs[0, 3].set_title(f"sqrt(alphas bar)")
    axs[0, 4].set_title(f"1 - (alphas bar)")

    plt.legend()
    if show_plot:
        plt.show()


if __name__ == '__main__':
    # visualize_noise_adding_methods()
    # visualize_full_noise()
    # visualize_dataset_sample('flowers')
    # visualize_noise_mean_variance_schedules()

    print('Uncomment one of the lines in the `visualize.py` file to what ?? get this... visualize something !!!')
