import matplotlib.pyplot as plt
import torch

from tqdm import tqdm

from beta_scheduler import SCHEDULE_METHODS, make_alpha_from_beta, make_beta_schedule
from forward_diffusion import forward_diffusion
from utils import image_from_path_to_tensor, show_tensor_image


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

        betas = make_beta_schedule(method, timestep_nbr=T, to_numpy=False)
        hyperparameters = make_alpha_from_beta(betas, to_numpy=False, device=device)
        print("Hyperparameters created")

        for idx in tqdm(range(0, num_images), unit='img'):
            timestep = idx*stepsize
            img, _ = forward_diffusion(
                image, timestep, hyperparameters['alphas_bar_sqrt'], hyperparameters['alphas_bar_one_minus']
            )
            show_tensor_image(img, ax=axs[index, idx])

        axs[index, 0].set_title(method)

    plt.legend()
    if show_plot:
        plt.show()

if __name__ == '__main__':
    visualize_noise_adding_methods()
