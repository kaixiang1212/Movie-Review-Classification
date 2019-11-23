from torchtext import data
from imdb_dataloader import IMDB
from collections import Counter
import torch.nn as tnn
import torch


class PreProcessing:
    def pre(x: list):
        """Experiment with your preprocessing here"""
        c = []
        # full remove
        # if match the whole word -> remove
        html_format = ['/><br', '<br', '\x96']

        # partially remove
        numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
        noise = ['<br', '/>']
        punctuation = ['/', '"', ',', ':', '(', ')', '<', '>', '?', '.', '-', '\\', '!', '=', '*']
        uninformatives = numbers + punctuation
        prune = ['the', 'a', 'and', 'this', 'that', 'of', 'to', 'in', 'was', 'as', 'with', 'as', 'it', 'for', 'but',
                 'on', 'you', 'he', 'his', '']
        for word in x:
            if word not in html_format:
                for n in noise:
                    word = word.replace(n, '')
                for u in uninformatives:
                    word = word.replace(u, '')
                if word not in prune:
                    c.append(word)
        return c

    text_field = data.Field(lower=True, include_lengths=True, batch_first=True, preprocessing=pre)


textField = PreProcessing.text_field
labelField = data.Field(sequential=False)
train = IMDB.splits(textField, labelField, train="train")[0]
freq = Counter()
for example in train.examples:
    freq.update(example.text)
print(freq.most_common(10))

# 83.7868%
# uninformatives = ['the', 'a', 'and', 'this', 'that', 'of', 'to', 'in', 'was', 'as', 'with', 'as', 'it', 'for', 'but',
#                   'on', 'you', 'he']
