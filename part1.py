#!/usr/bin/env python3
"""
part1.py

UNSW COMP9444 Neural Networks and Deep Learning

ONLY COMPLETE METHODS AND CLASSES MARKED "TODO".

DO NOT MODIFY IMPORTS. DO NOT ADD EXTRA FUNCTIONS.
DO NOT MODIFY EXISTING FUNCTION SIGNATURES.
DO NOT IMPORT ADDITIONAL LIBRARIES.
DOING SO MAY CAUSE YOUR CODE TO FAIL AUTOMATED TESTING.
"""

import torch


class rnn(torch.nn.Module):

    def __init__(self):
        super(rnn, self).__init__()

        self.ih = torch.nn.Linear(64, 128)
        self.hh = torch.nn.Linear(128, 128)

    def rnnCell(self, input, hidden):
        # Passed
        return torch.tanh(self.ih(input) + self.hh(hidden))

    def forward(self, input):
        hidden = torch.zeros(128)
        # Passed
        for i in range(input.size(0)):
            hidden = self.rnnCell(input[i], hidden)

        return hidden


class rnnSimplified(torch.nn.Module):

    def __init__(self):
        super(rnnSimplified, self).__init__()
        """
        TODO: Define self.net using a single PyTorch module such that
              the network defined by this class is equivalent to the
              one defined in class "rnn".
        """
        self.net = torch.nn.RNN(input_size=64, hidden_size=128)

    def forward(self, input):
        _, hidden = self.net(input)

        return hidden


def lstm(input, hiddenSize):
    """
    TODO: Let variable lstm be an instance of torch.nn.LSTM.
          Variable input is of size [batchSize, seqLength, inputDim]
    """
    lstm = torch.nn.LSTM(input_size=input.size(2), hidden_size=hiddenSize, batch_first = True)
    return lstm(input)


def conv(input, weight):
    # Passed
    return torch.nn.functional.conv1d(input, weight)
