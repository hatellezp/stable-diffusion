import torch
import torch.nn.functional as F

from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from forward_diffusion import forward_diffusion
from utils import ArrayOrTensor


def compute_l1_loss(
        model: nn.Module,
        x0: torch.Tensor,
        timestep: int | torch.Tensor,
        alphas_bar_sqrt: ArrayOrTensor,
        alpha_bar_one_minus: ArrayOrTensor) -> torch.Tensor:

    x_noisy, noise = forward_diffusion(x0, timestep, alphas_bar_sqrt, alpha_bar_one_minus)
    noise_pred = model(x_noisy, timestep)

    return F.l1_loss(noise, noise_pred)


@torch.no_grad()  # why torch no grad ??
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """

    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

def default_optimizer(model: nn.Module) -> Optimizer:
    return Adam(model.parameters(), lr=0.001)


def train(model: nn.Module, dataset: torch.utils.data.Dataset, batch_size: int, epochs: int, optimizer=None):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = default_optimizer(model) if optimizer is None else optimizer
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)



    for epoch in range(epochs):
        for step, batch in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()

            t = torch.randint(0, T, (batch_size,), device=device).long()
            loss = get_loss(model, batch[0], t)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 and step == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                # sample_plot_image()
