# %%
import numpy as np
import sklearn.metrics
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import os

#%%
model = models.resnet18(pretrained=True)


# %%
transform = transforms.Compose([
    transforms.Grayscale(3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
mnist = datasets.MNIST('./var', download=True)

train = datasets.MNIST('./var', train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    train, batch_size=512, shuffle=True, num_workers=8)
test = datasets.MNIST('./var', train=False, transform=transform)
testloader = torch.utils.data.DataLoader(
    test, batch_size=512, shuffle=True, num_workers=8)
for inputs, outputs in trainloader:
    # slice out one channel
    image = inputs[0][0]
    plt.imshow(image.numpy(), cmap=plt.get_cmap('binary'))
    break



# %%
model.fc = torch.nn.Linear(model.fc.in_features, 10)


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
for epoch in range(1):
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

#%%
print(sklearn.metrics.accuracy_score(
    np.concatenate(actual_buffer),
    np.concatenate(results_buffer)))