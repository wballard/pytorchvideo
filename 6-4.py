
# %%
import numpy as np
import spacy
import tqdm
from torch.utils.data import Dataset
import pandas
import torch
import sklearn.metrics
import torch.utils.data

# %%[markdown]
That one hot + neural network model did not work very well!
So now for a different
technique that treats text as a sequence, this will involve recurrent
networks, using a particular kind called an LSTM.

# %%
nlp = spacy.load('en_core_web_lg')

# %%[markdown]
Using the same dataset of sentiments on movie reviews, we
will use a pre-trained language model from spacy.

Using wikipedia, spacy comes pretrained with word vectors, which
are dense encodings - -so instead of one hot encoding, we use
the word vector.

The nice thing about this is we actually do less work to set up
our data - -AND - -our model starts with knowledge from the language
model built over wikipedia.

Here is an example word vector:

#%%
for token in nlp('hello'):
    print(token)
    print(token.vector)
    print(token.vector.shape)


# %%[markdown]
And for the dataset, we just extract the vectors as tensors
and return the length of each string in tokens.
This is important for working with pytorch recurrent networks.

# %%
class SentimentDataset(Dataset):
    def __init__(self):
        self.data = pandas \
            .read_csv('sentiment.tsv', sep='\t', header=0) \
            .groupby('SentenceId') \
            .first()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if type(idx) is torch.Tensor:
            idx = idx.item()
        sample = self.data.iloc[idx]
        token_vectors = []
        # switching off NER for a tiny speed boost
        for token in nlp(sample.Phrase.lower(), disable=['ner']):
            token_vectors.append(token.vector)

        # tokens and length as inputs -- the length
        # is needed to 'pack' variable length sequences
        # output is the sentiment score 
        return (torch.tensor(token_vectors),
                torch.tensor(len(token_vectors)),
                torch.tensor(sample.Sentiment, dtype=torch.float))


sentiment = SentimentDataset()
sentiment[0]


# %%
# break this into a training and testing dataset, and need
# to collate into fixed width as these will be
# variable batches
def collate(batch):
    # sort indescending length order -- this is needed for
    # padding seqeunces in pytorch
    batch.sort(key=lambda x: x[1], reverse=True)
    sequences, lengths, sentiments = zip(*batch)
    sequences = torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=True
    )
    sentiments = torch.stack(sentiments)
    lengths = torch.stack(lengths)
    return sequences, lengths, sentiments


number_for_testing = int(len(sentiment) * 0.05)
number_for_training = len(sentiment) - number_for_testing
train, test = torch.utils.data.random_split(sentiment,
                                            [number_for_training, number_for_testing])
trainloader = torch.utils.data.DataLoader(
    train, batch_size=32, shuffle=True,
    num_workers=4,
    collate_fn=collate)
testloader = torch.utils.data.DataLoader(
    test, batch_size=32, shuffle=True,
    collate_fn=collate)

# take a peek and see what we are collating
for batch in trainloader:
    print(batch[0].shape, batch[1].shape, batch[2].shape)
    # what is the max length?
    print(batch[1][0])
    break

# %% [markdown]
Now, this is still a regression problem, but instead of one hot
encoded words and a plain nerual network, we will have sequences
of word vectors, from the learned wikipedia model.

These sequences in turn will be * packed * which is because they
all have different lengths, run through the recurrent network
which loops word vector by word vector to compute a final
numerical representation of the whole sequence - -just like
reading - -word for word in order.

This is why we need the sequence lengths - -you need to know
the boundaries on which to pack.

# Now we have a regression problem! We 'll try to predict the sentiment
# using our approach for simple regression from earlier videos.


# %%
class Model(torch.nn.Module):

    def __init__(self, input_dimensions, size=128, layers=2):
        super().__init__()
        self.seq = torch.nn.LSTM(input_dimensions, size, layers)
        self.norm = torch.nn.LayerNorm(size * layers)
        self.layer_one = torch.nn.Linear(size * layers, size)
        self.activation_one = torch.nn.ReLU()
        self.layer_two = torch.nn.Linear(size, size)
        self.activation_two = torch.nn.ReLU()
        self.shape_outputs = torch.nn.Linear(size, 1)

    def forward(self, inputs, lengths):
        # need to sort the sequences for pytorch -- which we
        # did in our collation above
        number_of_batches = lengths.shape[0]
        packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(
            inputs,
            lengths,
            batch_first=True)
        buffer, (hidden, cell) = self.seq(packed_inputs)
        # flatten out the last hidden state -- this will
        # be the tensor representing each batch
        buffer = hidden.view(number_of_batches, -1)
        # and feed along to a simple output network with
        # a single output cell for regression
        buffer = self.norm(buffer)
        buffer = self.layer_one(buffer)
        buffer = self.activation_one(buffer)
        buffer = self.layer_two(buffer)
        buffer = self.activation_two(buffer)
        buffer = self.shape_outputs(buffer)
        return buffer


# get the input dimensions from the first sample
# encodings are word, vectors - so index 1 at the end
model = Model(sentiment[0][0].shape[1])


# %%
optimizer = torch.optim.Adam(model.parameters())
loss_function = torch.nn.MSELoss()
model.train()
for epoch in range(16):
    losses = []
    for sequences, lengths, sentiments in tqdm.tqdm(trainloader):
        optimizer.zero_grad()
        results = model(sequences, lengths)
        loss = loss_function(results, sentiments)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    print("Loss: {0}".format(torch.tensor(losses).mean()))

# %% [markdown]
And now - -we can score the results and see what the R2 values
look like in comparison to the basic neural network model.


# %%
results_buffer = []
actual_buffer = []
with torch.no_grad():
    model.eval()
    for inputs, lengths, outputs in testloader:
        results = model(inputs, lengths).detach().numpy()
        actual = outputs.numpy()
        results_buffer.append(results)
        actual_buffer.append(actual)


print(sklearn.metrics.r2_score(
    np.concatenate(actual_buffer),
    np.concatenate(results_buffer)))
