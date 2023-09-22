import torch
import hydra

from omegaconf import DictConfig

from src.beta_scheduler import make_beta_schedule, make_alpha_from_beta
from src.inference import plot_image_sample
from src.model import SimpleUnet


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg : DictConfig) -> None:

    IMG_SIZE = cfg['img_size']
    IMG_CHANNELS = cfg['img_channels']
    UNET_SCHEDULE = tuple(cfg['unet_schedule'])
    CHECKPOINTS_PATH = cfg['checkpoints_path']
    T = cfg['total_noising_steps']
    MODEL_NAME = cfg['model_name']

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    # 2. Define a mean and variance schedule
    betas = make_beta_schedule('clipped_cosine', T, device=DEVICE)
    meanvar_schedule = make_alpha_from_beta(betas, device=DEVICE)

    # 3. Load the model
    model_path = CHECKPOINTS_PATH + '/' + MODEL_NAME
    print(f"Loading model from {model_path}")
    unet_model = SimpleUnet.from_pretrained(model_path, schedule=UNET_SCHEDULE, image_channels=IMG_CHANNELS, device=DEVICE).to(DEVICE)

    total_params = sum(p.numel() for p in unet_model.parameters())
    print(f"Using model SimpleUnet with {total_params} parameters.")

    unet_model.eval()
    plot_image_sample(unet_model, meanvar_schedule, IMG_SIZE, IMG_CHANNELS, T, device=DEVICE)


if __name__ == '__main__':
    main()
