import torch
import hydra

from omegaconf import DictConfig

from src.beta_scheduler import make_beta_schedule, make_alpha_from_beta
from src.model import SimpleUnet
from src.trainer import train, mse_loss
from src.utils import default_image_to_tensor_transform, gather_data


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg : DictConfig) -> None:

    DATASET = cfg['dataset']
    BATCH_SIZE = cfg['batch_size']
    IMG_SIZE = cfg['img_size']
    IMG_CHANNELS = cfg['img_channels']
    UNET_SCHEDULE = tuple(cfg['unet_schedule'])
    EPOCHS = cfg['epochs']
    DATASETS_PATH = cfg['datasets_path']
    CHECKPOINTS_PATH = cfg['checkpoints_path']
    T = cfg['total_noising_steps']
    MODEL_NAME = cfg['model_name']

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Gather data
    transform_image_to_tensor = default_image_to_tensor_transform(IMG_SIZE, IMG_SIZE)
    dataset = gather_data(DATASET, transform=transform_image_to_tensor)


    # 2. Define a mean and variance schedule
    betas = make_beta_schedule('clipped_cosine', T, device=DEVICE)
    meanvar_schedule = make_alpha_from_beta(betas, device=DEVICE)

    # 3. Create your model
    unet_model = SimpleUnet(schedule=UNET_SCHEDULE, image_channels=IMG_CHANNELS)
    unet_model = unet_model.to(DEVICE)
    total_params = sum(p.numel() for p in unet_model.parameters())
    print(f"Using model SimpleUnet with {total_params} parameters.")


    # 4. Train baby !
    train(
        unet_model,
        dataset,
        meanvar_schedule,
        BATCH_SIZE,
        EPOCHS,
        mse_loss,
        optimizer=None,
        device=DEVICE
    )

    # 5. Save the model
    model_path = CHECKPOINTS_PATH + '/' + MODEL_NAME
    unet_model.save_state_dict(model_path)
    print(f"Model saved to {model_path}")


if __name__ == '__main__':
    main()