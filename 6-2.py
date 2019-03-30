#%%
import spacy
import torch
# %%[markdown]
My favorite way to parse these days - -at least when I 'm working
with words - -is spacy, the Dockerfile includes english support, so
let 's parse!

# %%
nlp = spacy.load('en')

# %%[markdown]
Now parse some words from a sentence.

# %%
parsed = nlp('Hello there my jupyter notebook!')
for token in parsed:
    print(token)

# %%[markdown]
Bag of words is like one hot encoding when we are talking about
single words - -and the first thing you need to do to encode
is turn words into ordinals.

# %%
ordinals = {}
for token in parsed:
    if token.text not in ordinals:
        ordinals[token.text] = len(ordinals)
ordinals

# %%[markdown]
And now one hot encoding is simply a matter of 'turning on' the
part of a tensor when a word matches - -let 's look at a single word.

# %%
word_one_hot = torch.zeros(len(ordinals))
word_one_hot[ordinals['there']] = 1
word_one_hot

# %%[markdown]
And here is the interesting part - -while language clearly has order
and is a sequence, it is often good enough to ignore this and encode
all the words into a vector - -this is the bag - of - words.All the words
in the bag lose their order, so we simply count the number of times a word
appears and encode that in a tensor.

# %%
sentence_bag_of_words = torch.zeros(len(ordinals))
for token in parsed:
    sentence_bag_of_words[ordinals[token.text]] += 1

sentence_bag_of_words

# %%[markdown]
Now this is a bit of a silly example, since all the words are turned on,
but when you apply this technique to a list of texts, each with difference words
you can get a working input model.

This may seem like a really gross simplification of language - -and it is
- -and we will look to more advanced methods very soon, but for now we will
try to classify text with just this really simple method.


