import torch

from src.beta_scheduler import make_beta_schedule, make_alpha_from_beta
from src.inference import generate_sample_from_noise, plot_image_sample
from src.model import SimpleUnet
from src.trainer import train, mse_loss
from src.utils import default_tensor_to_image_transform, default_image_to_tensor_transform, gather_data

# Set parameters
DATASET = 'flowers'
BATCH_SIZE = 32
IMG_SIZE = 128
UNET_SCHEDULE = (128, 256, 512, 256, 128)
EPOCHS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASETS_PATH = "../../../Datasets"
CHECKPOINTS_PATH = "../../../Checkpoints"
T = 200


# 2. Define a mean and variance schedule
betas = make_beta_schedule('clipped_cosine', T, device=DEVICE)
hyperparameters_schedule = make_alpha_from_beta(betas, device=DEVICE)


# 3. Load a model
model_name = 'simpleUnet1'
model_path = CHECKPOINTS_PATH + '/' + model_name
unet_model = SimpleUnet.from_pretrained(model_path, schedule=UNET_SCHEDULE, device=DEVICE)
unet_model = unet_model.to(DEVICE)


# Allocate unused memory
if DEVICE == 'cuda':
    torch.cuda.empty_cache()

plot_image_sample(unet_model, hyperparameters_schedule, IMG_SIZE, T, device=DEVICE)
