
# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import tqdm
import numpy as np
import torchvision
from torchvision import models, datasets, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# %%

class Generator(nn.Module):
    def __init__(self, context, features, channels):
        super().__init__()
        self.main = nn.Sequential(
            # input is context vector
            nn.ConvTranspose2d(context, features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            # state size. (features*8) x 4 x 4 -- this is a 'reverse'
            nn.ConvTranspose2d(features * 8, features * \
                               4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # state size. (features*4) x 8 x 8
            nn.ConvTranspose2d(features * 4, features * \
                               2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            # state size. (features*2) x 16 x 16
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            # state size. (features) x 32 x 32
            nn.ConvTranspose2d(features, channels, 4, 2, 1, bias=False),
            # outputs a pixel value centered on 0
            nn.Tanh()
            # state size. (channels) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# %%


class Discriminator(nn.Module):
    def __init__(self, features, channels):
        super().__init__()
        self.main = nn.Sequential(
            # input is (channels) x 64 x 64
            nn.Conv2d(channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features) x 32 x 32
            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features*2) x 16 x 16
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features*4) x 8 x 8
            nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features*8) x 4 x 4
            nn.Conv2d(features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# %%[markdown]
And there is one other trick - -custom weight initialization.This is from
the original DCGAN paper.Subclass the models and have the initial weights
and parameters set with this distribution to help training.

#%%


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class GeneratorCustom(Generator):
    def __init__(self, context, features, channels):
        super.__init__(context, features, channels)
        weights_init(self)


class DiscriminatorCustom(Discriminator):
    def __init__(self, features, channels):
        super.__init__(features, channels)
        weights_init(self)


# %%[markdown]
And now for training,
we will work on our good friends the MNIST digits.

# %%
batch_size = 128

transform = transforms.Compose([
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
mnist = datasets.MNIST('./var', download=True)

real = datasets.MNIST('./var', train=True, transform=transform)
realloader = torch.utils.data.DataLoader(
    real, batch_size=batch_size, shuffle=True)
for inputs, outputs in realloader:
    # slice out one channel
    image = inputs[0][0]
    plt.imshow(image.numpy(), cmap=plt.get_cmap('binary'))
    break


# %%[markdown]
And here is the training loop - -this is a bit different in that it
makes multiple passes and really shows off the flexibility of pytorch.

First we will take a real batch, and train

# %%
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#######
# Hyperparameters!

epochs = 16 
# probability labels -- all real or all fake
real_label = 1
fake_label = 0

# our parameters
context_size = 10
features = 32
channels = 1

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
#######

# binary cross entropy -- this is great comparing
# single probabilities, such as a single 'real' and a
# single 'fake'
criterion = nn.BCELoss()

# this is a random number generator to be used as a see
# to visualize how well we are doing!
fixed_noise = torch.randn(features, context_size, 1, 1, device=device)

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []

netD = Discriminator(features, channels).to(device)
# why 10? it should easily represent the 10 digits as a
# distribution
netG = Generator(context_size, features, channels).to(device)


# Setup Adam optimizers for both G and D
optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


#%%
# ok -- what does this look like at initialization?
fake = netG(fixed_noise).detach().cpu()
samples = vutils.make_grid(fake, padding=2, normalize=True)
plt.axes().imshow(samples.permute(1, 2, 0))


#%%

for epoch in range(epochs):
    # For each batch in the dataloader
    with tqdm.tqdm(realloader, unit='batches') as progress:
        for i, (data, _) in enumerate(realloader):
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            # Train with all-real batch
            netD.zero_grad()
            # Format batch
            batch_size = data.shape[0]
            real_data = data.to(device)
            real_labels = torch.full((batch_size,), real_label, device=device)
            # Forward pass real batch through D, and flatten it
            output = netD(real_data).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, real_labels)
            # Calculate gradients for D in backward pass
            errD_real.backward()

            # Train with all-fake batch
            # Generate a fresh fake batch
            noise = torch.randn(batch_size, context_size, 1, 1, device=device)
            fake_labels = torch.full((batch_size,), fake_label, device=device)
            # Generate fake image batch with G
            fake_data = netG(noise)
            # Classify all fake batch with D
            output = netD(fake_data).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, fake_labels)
            # Calculate the gradients for this batch
            errD_fake.backward(retain_graph=True)
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            # (2) Update G network: maximize log(D(G(z)))
            netG.zero_grad()
            # Calculate G's loss based on this output
            # this is the 'trick' if there is one -- fake data
            # through the Discriminator -- then how well the generator
            # fools the discriminator - but we keep the same tensor
            fake_labels.fill_(real_label)
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake_data).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, real_labels)
            # Calculate gradients for G
            # this has the effect of pusing both real and fake label
            # testing through the discriminator and the generator
            # via the linkage of the fake data
            errG.backward()
            # Update G
            optimizerG.step()

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Output training stats
            progress.set_postfix(
                G_loss=torch.tensor(G_losses).mean(),
                D_loss=torch.tensor(D_losses).mean(),
                refresh=False)
            progress.update()

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()

        samples = vutils.make_grid(fake, padding=2, normalize=True)
        img_list.append(samples)
#%%
#%%
ims = plt.axes().imshow(img_list[0].permute(1, 2, 0))

#%%
ims = plt.axes().imshow(img_list[-1].permute(1, 2, 0))
