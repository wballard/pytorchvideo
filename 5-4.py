# %%
import numpy as np
import sklearn.metrics
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import os

# %% [markdown]
Same as with VGG - - let's load up the pre-trained model.

# %%
model = models.resnet18(pretrained=True)

# %% [markdown]
We can drill in the code here and take a look at what makes ResNet tick.

# %%
model
# %%
model.fc

# %% [markdown]
Similar to with VGG - - we 'll need to adjust the images
to fit the pretraining expectations.

Trained on color images - - so we 'll need to expand
to three color channels.


We 'll do something simple and use MNIST with 10 classes to illustrate
the changes that need to be made.

# %%
transform = transforms.Compose([
    transforms.Grayscale(3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
mnist = datasets.MNIST('./var', download=True)

train = datasets.MNIST('./var', train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    train, batch_size=32, shuffle=True)
test = datasets.MNIST('./var', train=False, transform=transform)
testloader = torch.utils.data.DataLoader(
    test, batch_size=32, shuffle=True)
for inputs, outputs in trainloader:
    # slice out one channel
    image = inputs[0][0]
    plt.imshow(image.numpy(), cmap=plt.get_cmap('binary'))
    break


# %% [markdown]
And adjusting the final classifier(fc!), it 's a bit different
with ResNet, which is why you need to dig in to the source model
implementation to see where you need to modify the model.

# %%
model.fc = torch.nn.Linear(model.fc.in_features, 10)

# %% [markdown]
Let 's try this on a GPU. And we' ll add in tqdm as a
progress bar - -this one is really nice since you can just
wrap an enumerable such as the data loader.

# %%
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# %%
model.to(device)
model.train()
for epoch in range(16):
    for inputs, outputs in tqdm(trainloader):
        inputs = inputs.to(device, non_blocking=True)
        outputs = outputs.to(device, non_blocking=True)
        optimizer.zero_grad()
        results = model(inputs)
        loss = loss_function(results, outputs)
        loss.backward()
        optimizer.step()
    print("Last loss: {0}".format(loss))


# %%
results_buffer = []
actual_buffer = []
with torch.no_grad():
    model.eval()
    for inputs, actual in testloader:
        inputs = inputs.to(device, non_blocking=True)
        results = model(inputs).argmax(dim=1).to('cpu').numpy()
        results_buffer.append(results)
        actual_buffer.append(actual)

print(sklearn.metrics.classification_report(
    np.concatenate(actual_buffer),
    np.concatenate(results_buffer)))
