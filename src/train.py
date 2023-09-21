import torch
import torch.nn.functional as F

from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict

from forward_diffusion import forward_diffusion
from utils import ArrayOrTensor, get_value_and_reshape


def compute_l1_loss(
        model: nn.Module,
        x0: torch.Tensor,
        timestep: IntOrTensor,
        alphas_bar_sqrt: ArrayOrTensor,
        alpha_bar_one_minus: ArrayOrTensor) -> torch.Tensor:

    x_noisy, noise = forward_diffusion(x0, timestep, alphas_bar_sqrt, alpha_bar_one_minus)
    noise_pred = model(x_noisy, timestep)

    return F.l1_loss(noise, noise_pred)


def default_optimizer(model: nn.Module) -> Optimizer:
    return Adam(model.parameters(), lr=0.001)


def train(
        model: nn.Module,
        dataset: torch.utils.data.Dataset,
        total_steps: int,
        batch_size: int,
        epochs: int,
        optimizer=None):

    # Note: the total steps MUST be the same value that generated the schedule !

    # 1. Prepare device, select optimizer and setup a dataloader.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    model.to(device)
    optimizer = default_optimizer(model) if optimizer is None else optimizer
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 2. Actual training.
    for epoch in range(epochs):
        for step, batch in enumerate(tqdm(dataloader)):
            # Reinitialize the gradient.
            optimizer.zero_grad()

            # 2.1 Select a random step (actually give it the shape of the batch.)
            t = torch.randint(0, total_steps, (batch_size,), device=device).long()

            # 2.2 Compute loss
            loss = get_loss(model, batch[0], t)

            # 2.3 Backpropagate loss
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 and step == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
