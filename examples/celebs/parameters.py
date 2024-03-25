"""Parameters for DCGAN Tutorial"""

from pathlib import Path

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results

dataroot = Path(r"C:\tmp\DCGAN\Examples")  # Root directory for dataset
workers = 2  # Number of workers for dataloader
batch_size = 128  # Batch size during training

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64
nc = 3  # Number of channels in the training images. For color images this is 3
nz = 100  # Size of z latent vector (i.e. size of generator input)
ngf = 64  # Size of feature maps in generator
ndf = 64  # Size of feature maps in discriminator
num_epochs = 5  # Number of training epochs
lr = 0.0002  # Learning rate for optimizers
beta1 = 0.5  # Beta1 hyperparameter for Adam optimizers
ngpu = 1  # Number of GPUs available. Use 0 for CPU mode.
