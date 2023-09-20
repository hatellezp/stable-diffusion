import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import torchvision
import torch

from torch.utils.data import DataLoader
from torchvision import transforms

from beta_scheduler import make_beta_schedule, make_alpha_from_beta
from forward_diffusion import forward_diffusion
from utils import show_images, show_tensor_image, make_full_noise_sample

def diffuse_image(image: np.ndarray,
                  timestep: int,
                  beta_schedule_type: Tuple[float, ...],
                  mean: Tuple[float, ...],
                  variance: Tuple[float, ...]) -> np.ndarray:
    pass


def f2():
    schedules = ['linear', 'cosine', 'sqrt_linear', 'sqrt','log', 'linear_cosine', 'log_cosine']
    fig, axs = plt.subplots(3, 5)
    ts = [100, 1000, 10000]

    for idx in range(len(ts)):
        T = ts[idx]
        l = list(range(1, T + 1))
        for schedule in schedules:
            betas = make_beta_schedule(schedule, timestep_nbr=T)
            alphas, alphas_bar, alphas_bar_sqrt, alphas_bar_one_minus = make_alpha_from_beta(betas)

            axs[idx, 0].plot(l, betas, label=schedule)
            axs[idx, 1].plot(l, alphas, label=schedule)
            axs[idx, 2].plot(l, alphas_bar, label=schedule)
            axs[idx, 3].plot(l, alphas_bar_sqrt, label=schedule)
            axs[idx, 4].plot(l, alphas_bar_one_minus, label=schedule)



        if idx > 0:
            axs[idx, 0].set_title(f"A time schedule with T={T}")

    axs[0, 0].set_title(f"A time schedule with T={ts[0]}.\nBetas, smaller means more noise.")
    axs[0, 1].set_title(f"1 - betas")
    axs[0, 2].set_title(f"alphas bar")
    axs[0, 3].set_title(f"sqrt(alphas bar)")
    axs[0, 4].set_title(f"1 - (alphas bar)")



    plt.legend()
    plt.show()


def f1():
    DATASETS_PATH = "../../../Datasets"

    data = torchvision.datasets.OxfordIIITPet(root=DATASETS_PATH, download=True)
    show_images(data)

def f3():
    noise = make_full_noise_sample((3, 32, 32))
    show_tensor_image(noise)


def f4():
    plt.figure(figsize=(15,15))
    plt.axis('off')

    num_images = 10
    T = 300
    stepsize = int(T/num_images)
    DATASETS_PATH = "../../../Datasets"
    IMG_SIZE = 128
    BATCH_SIZE = 64

    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)
    data = torchvision.datasets.OxfordIIITPet(root=DATASETS_PATH, download=True, transform=data_transform)
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    x0 = next(iter(dataloader))[0]




    betas = make_beta_schedule('cosine', timestep_nbr=T, to_numpy=False)
    alphas, alphas_bar, alphas_bar_sqrt, alphas_bar_one_minus = make_alpha_from_beta(betas, to_numpy=False)




    for idx in range(0, T, stepsize):
        t = torch.Tensor([idx]).type(torch.int64)
        plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
        img, noise = forward_diffusion(x0, idx, alphas_bar_sqrt, alphas_bar_one_minus)
        show_tensor_image(img)
    plt.show()




if __name__ == '__main__':
    # f1()
    # f2()
    # f3()
    f4()