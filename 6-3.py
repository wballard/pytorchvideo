# %%
import torch.utils.data
import sklearn.metrics
import torch
import pandas
from torch.utils.data import Dataset
import tqdm
import spacy
import numpy as np
nlp = spacy.load('en')

# %%
pandas.read_csv('sentiment.tsv', sep='\t', header=0).iloc[0]


#%%[markdown]
OK - -now we will use one hot encoding to represent words as bitmaps
which are effectively 'word pixels' - but count the number of
times each word appears -- this is the basic bag of words model

# %%


class SentimentDataset(Dataset):
    def __init__(self):
        self.data = pandas \
            .read_csv('sentiment.tsv', sep='\t', header=0) \
            .groupby('SentenceId') \
            .first()
        # one hot encoding prep
        self.ordinals = {}
        for sample in tqdm.tqdm(self.data.Phrase):
            for token in nlp(sample.lower(), disable=['parser', 'tagger', 'ner']):
                if token.text not in self.ordinals:
                    self.ordinals[token.text] = len(self.ordinals)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if type(idx) is torch.Tensor:
            idx = idx.item()
        sample = self.data.iloc[idx]
        bag_of_words = torch.zeros(len(self.ordinals))
        for token in nlp(sample.Phrase.lower(), disable=['parser', 'tagger', 'ner']):
            bag_of_words[self.ordinals[token.text]] += 1

        return bag_of_words, torch.tensor(sample.Sentiment)


sentiment = SentimentDataset()

# %%
sentiment[0]


# %%[markdown]
Let us take a look at how many features we are really talking about.With words
encoded as one hot - -it is somewhat like a pixel, and the sentence is somewhat
like a bitmap.

# %%
len(sentiment.ordinals)


# %%
# break this into a training and testing dataset
number_for_testing = int(len(sentiment) * 0.05)
number_for_training = len(sentiment) - number_for_testing
train, test = torch.utils.data.random_split(sentiment,
                                            [number_for_training, number_for_testing])
trainloader = torch.utils.data.DataLoader(
    train, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(
    test, batch_size=32, shuffle=True)

len(test), len(train)

#%%
# and let's take a look at the output values
sentiment.data.Sentiment.unique()

# %%
# Now we have a simple classification problem with 5 classes
# let's use the most basic network possible and see what happens

# %%
class Model(torch.nn.Module):

    def __init__(self, input_dimensions, size=128):
        super().__init__()
        self.layer_one = torch.nn.Linear(input_dimensions, size)
        self.activation_one = torch.nn.ReLU()
        self.layer_two = torch.nn.Linear(size, size)
        self.activation_two = torch.nn.ReLU()
        self.shape_outputs = torch.nn.Linear(size, 5)

    def forward(self, inputs):
        buffer = self.layer_one(inputs)
        buffer = self.activation_one(buffer)
        buffer = self.layer_two(buffer)
        buffer = self.activation_two(buffer)
        buffer = self.shape_outputs(buffer)
        return buffer


model = Model(len(sentiment.ordinals))


# %%
optimizer = torch.optim.Adam(model.parameters())
loss_function = torch.nn.CrossEntropyLoss()
model.train()
for epoch in range(16):
    losses = []
    for inputs, outputs in tqdm.tqdm(trainloader):
        optimizer.zero_grad()
        results = model(inputs)
        loss = loss_function(results, outputs)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    print("Loss: {0}".format(torch.tensor(losses).mean()))

#%%[markdown]
And now a classificartion report to see how well we did
in recall and accuracy.

# %%
results_buffer = []
actual_buffer = []
with torch.no_grad():
    model.eval()
    for inputs, actual in testloader:
        results = model(inputs).argmax(dim=1).numpy()
        results_buffer.append(results)
        actual_buffer.append(actual)

print(sklearn.metrics.classification_report(
    np.concatenate(actual_buffer),
    np.concatenate(results_buffer)))
