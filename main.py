import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import torchvision
import torch

from torch.utils.data import DataLoader
from torchvision import transforms

from beta_scheduler import make_beta_schedule, make_alpha_from_beta, SCHEDULE_METHODS

def f2():
    schedules = ['linear', 'cosine', 'sqrt_linear', 'sqrt','log', 'linear_cosine', 'log_cosine']
    fig, axs = plt.subplots(3, 5)
    ts = [100, 1000, 10000]

    for idx in range(len(ts)):
        T = ts[idx]
        l = list(range(1, T + 1))
        for schedule in schedules:
            betas = make_beta_schedule(schedule, timestep_nbr=T)
            hyperparameters = make_alpha_from_beta(betas)

            axs[idx, 0].plot(l, betas, label=schedule)
            axs[idx, 1].plot(l, hyperparameters['alphas'], label=schedule)
            axs[idx, 2].plot(l, hyperparameters['alphas_bar'], label=schedule)
            axs[idx, 3].plot(l, hyperparameters['alphas_bar_sqrt'], label=schedule)
            axs[idx, 4].plot(l, hyperparameters['alphas_bar_one_minus'], label=schedule)



        if idx > 0:
            axs[idx, 0].set_title(f"A time schedule with T={T}")

    axs[0, 0].set_title(f"A time schedule with T={ts[0]}.\nBetas, smaller means more noise.")
    axs[0, 1].set_title(f"1 - betas")
    axs[0, 2].set_title(f"alphas bar")
    axs[0, 3].set_title(f"sqrt(alphas bar)")
    axs[0, 4].set_title(f"1 - (alphas bar)")



    plt.legend()
    plt.show()


if __name__ == '__main__':
    print('bob')