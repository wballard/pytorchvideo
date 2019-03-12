# %%
import sklearn.metrics
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

# %% [markdown]
The first thing to do - -load up a pretrained model.

Torchvision makes this pretty easy.

# %%
vgg = models.vgg11_bn(pretrained=True)

# %% [markdown]
Now, using what we learned by looking at the implementation of
VGG, let 's look under the hood at the feature and classifer
components.

# %%
vgg.features
# %%
vgg.classifier

# %% [markdown]
Now, the important thing to know - -is the input, we need to
input images - -one thing to just know, this VGG model was
trained with 224 x224 square images on image net - - 1000 classes.

And - -VGG was trained on color images - -so we 'll need to expand
to three color channels.

We 'll do something simple and use MNIST with 10 classes to illustrate
the changes that need to be made.

# %%
transform = transforms.Compose([
    transforms.Grayscale(3),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
])
mnist = datasets.MNIST('./var', download=True)

train = datasets.MNIST('./var', train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
test = datasets.MNIST('./var', train=False, transform=transform)
testloader = torch.utils.data.DataLoader(
    test, batch_size=32, shuffle=True)
for inputs, outputs in trainloader:
    # slice out one channel
    image = inputs[0][0]
    plt.imshow(image.numpy(), cmap=plt.get_cmap('binary'))
    break


# %% [markdown]
So there is an MNIST digit - -just a lot larger than we are used to
seeing.That 's the input - make sure you get the image and data
to the correct size - -now we turn to output.MNIST has 10 classes,
so we need to tweak the final output layer - - which will be what is
really being trained - - to 10 classes.

#%%
vgg.classifier[-1]


#%%
vgg.classifier[-1] = torch.nn.Linear(4096, 10)


#%% [markdown]
Let 's try this on a GPU. And we' ll add in tqdm as a
progress bar - -this one is really nice since you can just
wrap an enumerable such as the data loader.

#%%
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg.parameters())


vgg.to(device)
for epoch in range(16):
    total_loss = 0
    for inputs, outputs in tqdm(trainloader):
        inputs = inputs.to(device)
        outputs = outputs.to(device)
        optimizer.zero_grad()
        results = vgg(inputs)
        loss = loss_function(results, outputs)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print("Loss: {0}".format(total_loss / len(trainloader)))


#%%
for inputs, actual in testloader:
    inputs = inputs.to(device)
    results = vgg(inputs).argmax(dim=1).to('cpu').numpy()
    accuracy = sklearn.metrics.accuracy_score(actual, results)
    print(accuracy)
    break

print(sklearn.metrics.classification_report(actual, results))
