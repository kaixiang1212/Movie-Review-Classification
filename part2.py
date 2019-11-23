#!/usr/bin/env python3
"""
part2.py
UNSW COMP9444 Neural Networks and Deep Learning
ONLY COMPLETE METHODS AND CLASSES MARKED "TODO".
DO NOT MODIFY IMPORTS. DO NOT ADD EXTRA FUNCTIONS.
DO NOT MODIFY EXISTING FUNCTION SIGNATURES.
DO NOT IMPORT ADDITIONAL LIBRARIES.
DOING SO MAY CAUSE YOUR CODE TO FAIL AUTOMATED TESTING.
YOU MAY MODIFY THE LINE net = NetworkLstm().to(device)
"""

import numpy as np

import torch
import torch.nn as tnn
import torch.optim as topti

from torchtext import data
from torchtext.vocab import GloVe


# Class for creating the neural network.
class NetworkLstm(tnn.Module):
    """
    Implement an LSTM-based network that accepts batched 50-d
    vectorized inputs, with the following structure:
    LSTM(hidden dim = 100) -> Linear(64) -> ReLu-> Linear(1)
    Assume batch-first ordering.
    Output should be 1d tensor of shape [batch_size].
    """


    def __init__(self):
        super(NetworkLstm, self).__init__()
        """
        TODO:
        Create and initialise weights and biases for the layers.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lstm = tnn.LSTM(input_size = 50, hidden_size = 100, batch_first= True)
        self.layers = torch.nn.Sequential(
            tnn.Linear(in_features=100, out_features=64).to(self.device),
            tnn.ReLU()
        ).to(self.device)  
        self.lin = tnn.Linear(in_features=64, out_features=1).to(self.device)		
    def forward(self, input, length):
        """
        DO NOT MODIFY FUNCTION SIGNATURE
        TODO:
        Create the forward pass through the network.
        """
        batch_num = input.shape[0]
        #print(input.size())
        #print(batch_num)
        #x = input.permute(0, 2, 1) #index 0 becomes index 0, index 2 becomes index 1 and index 1 becomes index 2
        x = input
        x, (c,n) = self.lstm(x)
        #print(c)	
        #print(c.size())
        c = c.reshape(batch_num, -1)
        #print(c.size())
        c = self.layers(c)
        #print(c.size())
        c = self.lin(c)
        #print(c.size())
        c = c.reshape(batch_num)
        #print("final c ")
        #print(c.size())
        return c 

# Class for creating the neural network.
class NetworkCnn(tnn.Module):
    """
    Implement a Convolutional Neural Network.
    All conv layers should be of the form:
    conv1d(channels=50, kernel size=8, padding=5)
    Conv -> ReLu -> maxpool(size=4) -> Conv -> ReLu -> maxpool(size=4) ->
    Conv -> ReLu -> maxpool over time (global pooling) -> Linear(1)
    The max pool over time operation refers to taking the
    maximum val from the entire output channel. See Kim et. al. 2014:
    https://www.aclweb.org/anthology/D14-1181/
    Assume batch-first ordering.
    Output should be 1d tensor of shape [batch_size].
    """

    def __init__(self):
        super(NetworkCnn, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.layers = torch.nn.Sequential(
            tnn.Conv1d(in_channels=50, out_channels=50, kernel_size=8, padding=5),
            tnn.ReLU(),
            tnn.MaxPool1d(kernel_size=4),
            tnn.Conv1d(in_channels=50, out_channels=50, kernel_size=8, padding=5),
            tnn.ReLU(),
            tnn.MaxPool1d(kernel_size=4),
            tnn.Conv1d(in_channels=50, out_channels=50, kernel_size=8, padding=5),
            tnn.ReLU(),
            tnn.AdaptiveMaxPool1d(output_size=1)
        ).to(self.device)
        self.linear = tnn.Linear(in_features=50, out_features=1).to(self.device)

    def forward(self, input, length):
        print("forwarding")
        batch_num = input.shape[0]
        x = input.permute(0, 2, 1)
        x = self.layers(x)
        x = x.reshape(batch_num, -1)
        x = self.linear(x)
        x = x.reshape(batch_num)
        return x


def lossFunc():
    return tnn.BCEWithLogitsLoss()


def measures(outputs, labels):
    # passed
    outputs = torch.sigmoid(outputs)
    tp, tn, fp, fn = 0, 0, 0, 0

    for i in range(len(labels)):
        if outputs[i] >= 0.5:
            if labels[i] == 1:
                tp += 1
            else:
                fp += 1
        elif outputs[i] < 0.5:
            if labels[i] == 1:
                fn += 1
            else:
                tn += 1
    return tp, tn, fp, fn


def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("Using device: " + str(device))

    # Load the training dataset, and create a data loader to generate a batch.
    textField = data.Field(lower=True, include_lengths=True, batch_first=True)
    labelField = data.Field(sequential=False)

    from imdb_dataloader import IMDB
    train, dev = IMDB.splits(textField, labelField, train="train", validation="dev")

    textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev)

    trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=64,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)

    # Create an instance of the network in memory (potentially GPU memory). Can change to NetworkCnn during development.
    net = NetworkCnn().to(device)
    # net = NetworkLstm().to(device)

    criterion = lossFunc()
    optimiser = topti.Adam(net.parameters(), lr=0.001)  # Minimise the loss using the Adam algorithm.

    # for epoch in range(10):
    for epoch in range(1):
        running_loss = 0

        for i, batch in enumerate(trainLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # PyTorch calculates gradients by accumulating contributions to them (useful for
            # RNNs).  Hence we must manually set them to zero before calculating them.
            optimiser.zero_grad()

            # Forward pass through the network.
            print("about to forward ma dude")
            output = net(inputs, length)

            loss = criterion(output, labels)
            # print(loss)

            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            running_loss += loss.item()

            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f" % (epoch + 1, i + 1, running_loss / 32))
                running_loss = 0

    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0

    # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
    # computations and reduce memory usage.
    with torch.no_grad():
        for batch in testLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            outputs = net(inputs, length)

            tp_batch, tn_batch, fp_batch, fn_batch = measures(outputs, labels)
            true_pos += tp_batch
            true_neg += tn_batch
            false_pos += fp_batch
            false_neg += fn_batch

    accuracy = 100 * (true_pos + true_neg) / len(dev)
    matthews = MCC(true_pos, true_neg, false_pos, false_neg)

    print("Classification accuracy: %.2f%%\n"
          "Matthews Correlation Coefficient: %.2f" % (accuracy, matthews))


# Matthews Correlation Coefficient calculation.
def MCC(tp, tn, fp, fn):
    numerator = tp * tn - fp * fn
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(numerator, denominator)


if __name__ == '__main__':
    main()