import torch

from src.beta_scheduler import make_beta_schedule, make_alpha_from_beta
from src.model import SimpleUnet
from src.trainer import train, mse_loss
from src.utils import default_image_to_tensor_transform, gather_data

# Set parameters
DATASET = 'oxford'
BATCH_SIZE = 32
IMG_SIZE = 128
UNET_SCHEDULE = (128, 256, 512, 256, 128)
EPOCHS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASETS_PATH = "../../../Datasets"
CHECKPOINTS_PATH = "../../../Checkpoints"
T = 200


# 1. Gather data
transform_image_to_tensor = default_image_to_tensor_transform(IMG_SIZE, IMG_SIZE)
dataset = gather_data(DATASET, transform=transform_image_to_tensor)


# 2. Define a mean and variance schedule
betas = make_beta_schedule('clipped_cosine', T, device=DEVICE)
hyperparameters_schedule = make_alpha_from_beta(betas, device=DEVICE)

# 3. Create your model
unet_model = SimpleUnet(schedule=UNET_SCHEDULE)
unet_model = unet_model.to(DEVICE)
total_params = sum(p.numel() for p in unet_model.parameters())
print(f"Using model SimpleUnet with {total_params} parameters.")

# 4. Train baby !
train(
    unet_model,
    dataset,
    hyperparameters_schedule,
    T,
    BATCH_SIZE,
    EPOCHS,
    mse_loss,
    optimizer=None,
    device=DEVICE
)

# 5. Save the model
model_name = 'simpleUnet1'
model_path = CHECKPOINTS_PATH + '/' + model_name
unet_model.save_state_dict(model_path)
