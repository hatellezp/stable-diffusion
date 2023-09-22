import torch
import torch.nn.functional as F

from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Callable

from .forward_diffusion import forward_diffusion_ddpm
from .tensor_types import ArrayOrTensor, IntOrTensor


# This should be made into a class...
def compute_loss_global(
        x0: torch.Tensor,
        model: nn.Module,
        timestep: torch.Tensor,
        meanvar: Dict[str, torch.Tensor],
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: str = None) -> torch.Tensor:

        alphas_bar_sqrt = meanvar['alphas_bar_sqrt']
        alphas_bar_one_minus = meanvar['alphas_bar_one_minus']

        x_noisy, noise = forward_diffusion_ddpm(x0, timestep, alphas_bar_sqrt, alphas_bar_one_minus, device=device)
        noise_pred = model(x_noisy, timestep)

        loss = loss_function(noise, noise_pred)
        return loss

def mse_loss(x0: torch.Tensor,
        model: nn.Module,
        timestep: torch.Tensor,
        meanvar: Dict[str, torch.Tensor],
        device: str = None)  -> torch.Tensor:

     return compute_loss_global(x0, model, timestep, meanvar, F.mse_loss, device=device)


def default_optimizer(model: nn.Module) -> Optimizer:
    return Adam(model.parameters(), lr=0.001)


def train(
        model: nn.Module,
        dataset: torch.utils.data.Dataset,
        meanvar_schedule: Dict[str, ArrayOrTensor],
        batch_size: int,
        epochs: int,
        loss_function,  # I need to find a way to give a coherent type to this
        optimizer: Optimizer = None,
        device: str = None):

    # Note: the total steps MUST be the same value that generated the schedule !
    # 0. Deduce total number of noising steps from the schedule
    total_steps = len(meanvar_schedule['betas'])

    # 1. Prepare device, select optimizer and setup a dataloader.
    model.train()
    if device is not None:
        model = model.to(device)
    optimizer = default_optimizer(model) if optimizer is None else optimizer
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"Training over {len(dataset)} samples, batch size: {batch_size}, iterations: {len(dataset) // batch_size}, noising steps: {total_steps}.")

    # 2. Actual training.
    for epoch in range(epochs):
        for step, batch in enumerate(tqdm(dataloader)):
            # 2.0 Reinitialize the gradient.
            optimizer.zero_grad()

            # 2.1 Select a random step (actually give it the shape of the batch.)
            timestep = torch.randint(0, total_steps, (batch_size,), device=device).long()

            # 2.2 Created noise and prediction
            x0 = batch[0]
            if device is not None:
                x0 = x0.to(device)

            # 2.3 Backpropagate loss
            loss = loss_function(x0, model, timestep, meanvar_schedule, device=device)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 and step == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")

    print(f"Last loss value: {loss.item()}")
