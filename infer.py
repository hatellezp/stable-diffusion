import torch
import hydra

from matplotlib import pyplot as plt
from omegaconf import DictConfig
from torch import nn

from src.beta_scheduler import make_beta_schedule, make_alpha_from_beta
from src.inference import plot_image_sample
from src.model import SimpleUnet, DDPM
from src.forward_diffusion import ForwardDiffusionDDPM
from src.loss_functions import DiffusionLoss
from src import utils


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg : DictConfig) -> None:

    IMG_SIZE = cfg['img_size']
    IMG_CHANNELS = cfg['img_channels']
    DATASET = cfg['dataset']
    UNET_SCHEDULE = tuple(cfg['unet_schedule'])
    CHECKPOINTS_PATH = cfg['checkpoints_path']
    T = cfg['total_noising_steps']
    MODEL_NAME = cfg['model_name']
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SKIP_STEPS = cfg['skip_steps']
    NBR_SAMPLES = cfg['number_of_samples_generated']

    # 1. Prepare the schedule
    beta_schedule = make_beta_schedule('cosine', T, device=DEVICE)
    meanvar_schedule = make_alpha_from_beta(beta_schedule, device=DEVICE)

    # 2. Prepare the model
    forward_diffusion_ddpm = ForwardDiffusionDDPM(meanvar_schedule, device=DEVICE)
    loss_function = DiffusionLoss(nn.functional.mse_loss, forward_diffusion_ddpm)

    simple_unet = SimpleUnet(IMG_CHANNELS, UNET_SCHEDULE, IMG_SIZE, device=DEVICE)
    ddpm = DDPM(simple_unet, loss_function, T).to(DEVICE)


    # 3. Load the model
    model_path = CHECKPOINTS_PATH + '/' + f'{DATASET}_{MODEL_NAME}_2'
    print(f"Loading model from {model_path}")
    ddpm.load_state_dict(torch.load(model_path))
    ddpm = ddpm.to(DEVICE)
    ddpm.eval()

    print(f'Denoising over {T} step(s), generating {NBR_SAMPLES} sample(s).')

    if NBR_SAMPLES == 1:
        xT = utils.make_full_noise_sample((1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)).to(DEVICE)

        result = ddpm.full_denoise(xT, device=DEVICE, step_size=SKIP_STEPS)
        utils.show_tensor_image(result.detach().cpu())
    else:
        plt.figure(figsize=(30,30))
        plt.axis('off')
        _, axs = plt.subplots(ncols=NBR_SAMPLES)

        for i in range(NBR_SAMPLES):
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()

            xT = utils.make_full_noise_sample((1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)).to(DEVICE)

            result = ddpm.full_denoise(xT, device=DEVICE, step_size=SKIP_STEPS)
            utils.show_tensor_image(result.detach().cpu(), ax=axs[i], show_plot=False)

        plt.show()


if __name__ == '__main__':
    main()
