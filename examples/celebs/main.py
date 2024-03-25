import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# from IPython.display import HTML

import parameters
import model
import training


def load_dataset() -> torch.utils.data.DataLoader:
    dataset = dset.ImageFolder(
        root=str(parameters.dataroot),
        transform=transforms.Compose(
            [
                transforms.Resize(parameters.image_size),
                transforms.CenterCrop(parameters.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=parameters.batch_size, shuffle=True, num_workers=parameters.workers
    )
    return dataloader


def plot_examples(dataloader: torch.utils.data.DataLoader, device: torch.device) -> None:
    """"""
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),
            (1, 2, 0),
        )
    )
    plt.show()


def initialize_model(device: torch.device) -> (model.Generator, model.Discriminator):
    """"""
    netG = model.Generator(ngpu=parameters.ngpu).to(device)

    # Handle multi-GPU if desired
    # if (device.type == 'cuda') and (parameters.ngpu > 1):
    #     netG = nn.DataParallel(netG, list(range(parameters.ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    netG.apply(model.weights_init)

    netD = model.Discriminator(ngpu=parameters.ngpu).to(device=device)
    netD.apply(model.weights_init)
    return netG, netD


def main():
    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and parameters.ngpu > 0) else "cpu"
    )
    print("Random Seed: ", parameters.manualSeed)
    random.seed(parameters.manualSeed)
    torch.manual_seed(parameters.manualSeed)
    torch.use_deterministic_algorithms(True)

    dataloader = load_dataset()
    # plot_examples(dataloader=dataloader, device=device)
    (netG, netD) = initialize_model(device=device)
    training.training_loop(netG=netG, netD=netD, dataloader=dataloader, device=device)

    print(netG)
    print(netD)
    print()


if __name__ == '__main__':
    main()
