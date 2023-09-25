import hydra
import lightning.pytorch as pl
import torch

from omegaconf import DictConfig
from torch import nn

from src import utils
from src.model import DDPM, SimpleUnet
from src.beta_scheduler import make_beta_schedule, make_alpha_from_beta
from src.utils import default_image_to_tensor_transform, gather_data, make_full_noise_sample
from src.loss_functions import DiffusionLoss
from src.forward_diffusion import ForwardDiffusionDDPM

# Use tensor cores
torch.set_float32_matmul_precision('high')


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

    # 1. Prepare the schedule
    beta_schedule = make_beta_schedule('cosine', T, device=DEVICE)
    meanvar_schedule = make_alpha_from_beta(beta_schedule, device=DEVICE)

    # 2. Gather data
    transform_image_to_tensor = default_image_to_tensor_transform(IMG_SIZE, IMG_SIZE)
    # train_dataset, test_dataset = gather_data(DATASET, dataset_path=DATASETS_PATH, transform=transform_image_to_tensor)
    train_dataset = gather_data(DATASET, dataset_path=DATASETS_PATH, transform=transform_image_to_tensor, separate_test_train=False)

    print(f'Shape of sample is {train_dataset[0][0].shape}')

    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=8, batch_size=BATCH_SIZE)
    # test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=8, batch_size=BATCH_SIZE)

    # 3. Prepare the model
    forward_diffusion_ddpm = ForwardDiffusionDDPM(meanvar_schedule, device=DEVICE)
    loss_function = DiffusionLoss(nn.functional.mse_loss, forward_diffusion_ddpm)

    simple_unet = SimpleUnet(IMG_CHANNELS, UNET_SCHEDULE, IMG_SIZE, device=DEVICE)
    ddpm = DDPM(simple_unet, loss_function, T).to(DEVICE)

    # 4. Prepare the trainer and fit
    trainer = pl.Trainer(limit_train_batches=100, max_epochs=EPOCHS)
    # trainer.fit(model=ddpm, train_dataloaders=train_loader, val_dataloaders=test_loader)
    trainer.fit(model=ddpm, train_dataloaders=train_loader)


    # 5. Validate

    # 6. Save the learned model
    path_to_parameters = CHECKPOINTS_PATH + '/' + f'{DATASET}_{MODEL_NAME}_2'
    ddpm.save_state_dict(path_to_parameters)
    print(f"Saved model state dict to {path_to_parameters}")


if __name__ == '__main__':
    main()
