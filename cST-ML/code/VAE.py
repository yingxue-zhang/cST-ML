import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torchvision import datasets
import matplotlib.pyplot as plt
import imageio
import itertools
import numpy as np
import struct
import copy

'''
class Encoder(nn.Module):
    r"""Encoder of VAE.

    Arguments:
        D_in: input channels of the input
        D_out: features numbers of output vector
        num_layers: lstm layer number
        z_dim: dimension of latant Gaussian
    """
    def __init__(self, D_in, D_out, num_layers, z_dim): # D_in = 4, batch = window_size-1
        super(Encoder, self).__init__()
        # input <- (batch, D_in, 5, 5).   output <- (batch, 64, 5, 5)
        self.conv1 = nn.Conv2d(in_channels=D_in, out_channels=64, kernel_size=3, stride=1, padding=1)
        # input <- (batch, 64, 5, 5).   output <- (batch, D_out, 1, 1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=D_out, kernel_size=5, stride=1, padding=0)
        self.linear1 = torch.nn.Linear(D_out, D_out)
        self.lstm = nn.LSTM(D_out, D_out, num_layers)
        self.linear2 = torch.nn.Linear(num_layers * D_out, D_out)

        self.mu = torch.nn.Linear(D_out, z_dim)
        self.log_var = torch.nn.Linear(D_out, z_dim)

    def forward(self, x):  # x is train data of a sub_task with x_train and y_train concatenated
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))

        x = x.view(x.size(0), 1, -1)
        out, (h, c) = self.lstm(x)

        h = h.view(-1)
        h = F.relu(self.linear2(h))

        mu = self.mu(h)
        log_var = self.log_var(h)
        return mu, log_var
'''

class Encoder(nn.Module):
    r"""Encoder of VAE.

    Arguments:
        D_in: input channels of the input
        D_out: features numbers of output vector
        num_layers: lstm layer number
        z_dim: dimension of latant Gaussian
    """
    def __init__(self, D_in, D_out, num_layers, z_dim): # D_in = 4, batch = window_size-1
        super(Encoder, self).__init__()
        # input <- (batch, D_in, 5, 5).   output <- (batch, 64, 5, 5)
        self.conv1 = nn.Conv2d(in_channels=D_in, out_channels=64, kernel_size=3, stride=1, padding=1)
        # input <- (batch, 64, 5, 5).   output <- (batch, D_out, 1, 1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=D_out, kernel_size=5, stride=1, padding=0)
        self.linear1 = torch.nn.Linear(D_out, D_out)
        self.lstm = nn.LSTM(num_layers * D_out + D_out, D_out, num_layers)
        self.linear2 = torch.nn.Linear(num_layers * D_out, D_out)

        self.mu = torch.nn.Linear(D_out, z_dim)
        self.log_var = torch.nn.Linear(D_out, z_dim)

    def forward(self, x, memory):  # x is train data of a sub_task with x_train and y_train concatenated
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))

        memory = memory.repeat(x.size(0), 1)
        x = torch.cat((x, memory), dim = 1)

        x = x.view(x.size(0), 1, -1)
        out, (h, c) = self.lstm(x)

        h = h.view(-1)
        h = F.relu(self.linear2(h))

        mu = self.mu(h)
        log_var = self.log_var(h)
        return mu, log_var, h


class Decoder(nn.Module):
    r"""Decoder of VAE.

    Arguments:
        D_in: input channels of the input
        D_out: features numbers of conv output
    """
    def __init__(self, D_in, D_out, z_dim):
        super(Decoder, self).__init__()
        # input <- (batch = 1, D_in, 5, 5).   output <- (batch = 1, 64, 5, 5)
        self.conv1 = nn.Conv2d(in_channels=D_in, out_channels=64, kernel_size=3, stride=1, padding=1)
        # input <- (batch = 1, 64, 5, 5).   output <- (batch = 1, D_out, 1, 1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=D_out, kernel_size=5, stride=1, padding=0)
        self.linear1 = torch.nn.Linear(D_out + z_dim, D_out)
        self.linear2 = torch.nn.Linear(D_out, 1)

    def forward(self, x, z):   # x is x_test of a sub_task, z is a z_sample
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(-1)
        out = torch.cat((x, z))

        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        return out


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample

    def forward(self, train_data, x_test, m):
        mu, log_var, memory = self.encoder(train_data, m)
        z = self.sampling(mu, log_var)
        return self.decoder(x_test, z), mu, log_var, memory
