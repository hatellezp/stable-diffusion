# Difussion Personal Implementation


THIS IS A WIP! Nothing is still of up to use or even read.

We are testing several diffusion methods:
- ddpm
- stable diffusion
- k-diffusion (to come)

## API

Just call `python train.py` for training and `python infer.py` to generate some samples.

### `config.yaml`
Both training and inference can be customized through the `config.yaml` file.

**Metadata**
- dataset: provide one of the datasets in `torchvision.data.datasets`
- datasets_path: where the datasets live
- checkpoints_path: same as above but for saved models

**Diffusion**
- unet_schedule: provide the ups and downs block parameters to build the unet that is at the core of the diffusion process defined here (you can go [here](https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/)for an example of implementation of a unet)
- total_denoising_steps: how many steps of noise are to be added to the sample
- skip_steps: steps to be skipped during the backward denoising processu (this can save memory during inference)


### Use your own data

You can of course train, fine-tune and test the architecture on your own data.
```python

    import torch

    from src.utils import *
    from src.forward_difussion import *
    from loss_functions import *
    from model import *

    batch_size = 16
    img_size = 32
    img_channels = 3
    epochs = 10
    total_noising_steps = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_path = 'mydatasets'

    # 1. Prepare the schedule
    beta_schedule = make_beta_schedule('cosine', total_noising_steps, device=device)
    meanvar_schedule = make_alpha_from_beta(beta_schedule, device=device)

    # 2. Gather the data: ddpm expects a torch Dataloder.
    train_data = torchvision.datasets.Flowers102(root=dataset_path, train=True, download=True, transform=toTensor()),
    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=8, batch_size=batch_size)

    # 3. Prepare the model
    forward_diffusion_ddpm = ForwardDiffusionDDPM(meanvar_schedule, device=device)
    loss_function = DiffusionLoss(nn.functional.mse_loss, forward_diffusion_ddpm)

    simple_unet = SimpleUnet(img_channels, img_size, device=device)
    ddpm = DDPM(simple_unet, loss_function, T).to(device)

    # 4. Prepare the trainer and fit
    trainer = pl.Trainer(limit_train_batches=100, max_epochs=epochs)
    # trainer.fit(model=ddpm, train_dataloaders=train_loader, val_dataloaders=test_loader)
    trainer.fit(model=ddpm, train_dataloaders=train_loader)

```


## Beta schedule issue:

One of the most important parts of diffusion models is to set a correct beta schedule.

- The initial proposition is a linear schedule, wich is (from intuition and experiments)
  to a good idea, indeed, at the end of the process the sample is fairly noisy and we
  keep adding a fair amount of unnecessary noise.
- The cosine noise discussed in this [paper](https://arxiv.org/pdf/2102.09672.pdf) tries
  to solve the problem above, but to no avail in my opinion.
  While for small `T` it seems the betas are smooth, for larger values of `T` like
  1000~10000 we can see a problem with the amount of noise added at the end, the function
  seems to almost lose its smoothness.

<img src="ressources/images/all_beta_schedules.png" />

For `alpha_bar` that is the cumulative product of `alpha` the courbe is more smooth in the
case of the cosine.
One temporary solution is to skip steps of diffusion. That is, skip some of the last steps
to clip the function of betas befor the end.

**TODO**: see what is more important in with respect to smoothness, `alpha` (equivalent to
`beta`) or `alpha_bar`


## Randomness of the guassian distribution and initial sample

The diffusion backward process begins with a full noise sample. I think that for, both
reproducibility of results and stability of the learning process is better to set a
global seed for the whole process.
