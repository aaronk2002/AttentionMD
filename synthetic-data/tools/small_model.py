import torch
from torch import nn


class AttWModel(nn.Module):
    """
    The model with the W parameterization
    """

    def __init__(self, input_size, std=0.01):
        """
        Create the model with dimensionality input_size
        """
        super(AttWModel, self).__init__()
        self.W = nn.Linear(input_size, input_size, bias=False)
        nn.init.normal_(self.W.weight, 0, std)
        self.v = nn.Parameter(torch.randn(input_size) * std)

    def forward(self, input_seq, cross_input):
        """
        Get the output of the model given the input sequence (X) and the
        cross input (z)
        """
        n, _, d = input_seq.shape
        cross_input = cross_input.reshape(n, 1, d)
        out = torch.softmax(cross_input @ self.W(input_seq).transpose(-2, -1), dim=-1)
        self.sfx_out = out
        out = out @ input_seq
        return out @ self.v


class AttKQModel(nn.Module):
    """
    The model with the KQ parameterization
    """

    def __init__(self, input_size, std=0.01):
        """
        Create the model with dimensionality input_size
        """
        super(AttKQModel, self).__init__()
        self.Q = nn.Linear(input_size, input_size, bias=False)
        self.K = nn.Linear(input_size, input_size, bias=False)
        nn.init.normal_(self.Q.weight, 0, 1)
        nn.init.normal_(self.K.weight, 0, std)
        self.v = nn.Parameter(torch.randn(input_size) * std)

    def forward(self, input_seq, cross_input):
        """
        Get the output of the model given the input sequence (X) and the
        cross input (z)
        """
        n, _, d = input_seq.shape
        cross_input = cross_input.reshape(n, 1, d)
        Q = self.Q(cross_input)
        K = self.K(input_seq)
        out = torch.softmax(Q @ K.transpose(-2, -1), dim=-1)
        self.sfx_out = out
        out = out @ input_seq
        return out @ self.v
