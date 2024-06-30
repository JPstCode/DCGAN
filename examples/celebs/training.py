import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from model import Generator
from model import Discriminator
import parameters


def training_loop(
    netG: Generator,
    netD: Discriminator,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
):
    """"""

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, parameters.nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.0
    fake_label = 0.0

    optimizerD = optim.Adam(netD.parameters(), lr=parameters.lr, betas=(parameters.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=parameters.lr, betas=(parameters.beta1, 0.999))

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(parameters.num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, parameters.nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print(
                    '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (
                        epoch,
                        parameters.num_epochs,
                        i,
                        len(dataloader),
                        errD.item(),
                        errG.item(),
                        D_x,
                        D_G_z1,
                        D_G_z2,
                    )
                )
                torch.save(netD.state_dict(), r"C:\tmp\DCGAN\models\shape\discriminator.pt")
                torch.save(netG.state_dict(), r"C:\tmp\DCGAN\models\shape\generator.pt")

                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()

                try:
                    plt.figure()
                    # Save fakes
                    for i in range(9):
                        fake_img = fake[i]
                        image_np = fake_img.numpy()
                        image_np = image_np.transpose(1, 2, 0)
                        plt.subplot(3, 3, i + 1)
                        plt.imshow(image_np)

                    plt.savefig(
                        fr"C:\tmp\DCGAN\models\shape\images\{epoch}.png"
                    )
                    plt.close()
                except Exception:
                    pass


            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or (
                (epoch == parameters.num_epochs - 1) and (i == len(dataloader) - 1)
            ):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    print()
