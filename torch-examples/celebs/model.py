"""GAN Model implementation."""

import torch.nn as nn

import parameters


# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(
                in_channels=parameters.nz,
                out_channels=parameters.ngf * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=parameters.ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(
                in_channels=parameters.ngf * 8,
                out_channels=parameters.ngf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=parameters.ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d(
                in_channels=parameters.ngf * 4,
                out_channels=parameters.ngf * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=parameters.ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d(
                in_channels=parameters.ngf * 2,
                out_channels=parameters.ngf,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=parameters.ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d(
                in_channels=parameters.ngf,
                out_channels=parameters.nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Tanh(),
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(
                in_channels=parameters.nc,
                out_channels=parameters.ndf,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(
                in_channels=parameters.ndf,
                out_channels=parameters.ndf * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=parameters.ndf * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(
                in_channels=parameters.ndf * 2,
                out_channels=parameters.ndf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=parameters.ndf * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(
                in_channels=parameters.ndf * 4,
                out_channels=parameters.ndf * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=parameters.ndf * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(
                in_channels=parameters.ndf * 8,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)
