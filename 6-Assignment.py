
# %%
import numpy as np
import spacy
import tqdm
from torch.utils.data import Dataset
import pandas
import torch
import sklearn.metrics
import torch.utils.data

# %%
nlp = spacy.load('en_core_web_lg')

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
                torch.tensor(sample.Sentiment))


sentiment = SentimentDataset()
sentiment[0]

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
    collate_fn=collate)
testloader = torch.utils.data.DataLoader(
    test, batch_size=32, shuffle=True,
    collate_fn=collate)

class Model(torch.nn.Module):

    def __init__(self, input_dimensions, size=32, layers=4):
        super().__init__()
        self.seq = torch.nn.GRU(input_dimensions, size, layers)
        self.layer_one = torch.nn.Linear(size * layers, size)
        self.activation_one = torch.nn.ReLU()
        self.layer_two = torch.nn.Linear(size, size)
        self.activation_two = torch.nn.ReLU()
        self.shape_outputs = torch.nn.Linear(size, 5)

    def forward(self, inputs, lengths):
        # need to sort the sequences for pytorch -- which we
        # did in our collation above
        number_of_batches = lengths.shape[0]
        packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(
            inputs,
            lengths,
            batch_first=True)
        buffer, hidden = self.seq(packed_inputs)
        # batch first...
        buffer = hidden.permute(1, 0, 2)
        # flatten out the last hidden state -- this will
        # be the tensor representing each batch
        buffer = buffer.contiguous().view(number_of_batches, -1)
        # and feed along to a simple output network with
        # a single output cell for regression
        buffer = self.layer_one(buffer)
        buffer = self.activation_one(buffer)
        buffer = self.layer_two(buffer)
        buffer = self.activation_two(buffer)
        buffer = self.shape_outputs(buffer)
        return buffer


model = Model(sentiment[0][0].shape[1])


# %%
optimizer = torch.optim.Adam(model.parameters())
loss_function = torch.nn.CrossEntropyLoss()
model.train()
for epoch in range(32):
    losses = []
    for sequences, lengths, sentiments in tqdm.tqdm(trainloader):
        optimizer.zero_grad()
        results = model(sequences, lengths)
        loss = loss_function(results, sentiments)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    print("Loss: {0}".format(torch.tensor(losses).mean()))


# %%
results_buffer = []
actual_buffer = []
with torch.no_grad():
    model.eval()
    for test_seq, test_len, test_sentiment in testloader:
        results = model(test_seq, test_len).argmax(dim=1).numpy()
        results_buffer.append(results)
        actual_buffer.append(test_sentiment)

print(sklearn.metrics.classification_report(
    np.concatenate(actual_buffer),
    np.concatenate(results_buffer)))
