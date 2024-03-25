""""""
import torch

import parameters

from matplotlib import pyplot as plt

from model import Generator


def plot_images(images):
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))  # Create a 4x4 grid of subplots

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Plot each image
    for i in range(16):
        image_np = images[i].cpu().numpy()
        image_np = image_np.transpose(1, 2, 0)
        ax = axes[i]
        ax.imshow(image_np)

        ax.axis('off')  # Turn off axis labels

    plt.tight_layout()  # Adjust layout to make sure images don't overlap
    plt.show()


def main():
    """"""

    device = torch.device("cuda:0" if (torch.cuda.is_available() and parameters.ngpu > 0) else "cpu")

    fixed_noise = torch.randn(64, parameters.nz, 1, 1, device=device)

    model = Generator(ngpu=parameters.ngpu).to(device=device)
    checkpoint = torch.load(r"C:\tmp\DCGAN\models\generator.pt", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    fakes = []
    with torch.no_grad():
        fakes = model(fixed_noise)

    plot_images(images=fakes)


if __name__ == '__main__':
    main()
