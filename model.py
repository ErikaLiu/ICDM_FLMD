import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt


"""Implementation of FLMD. 
Modified by Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
"""


class VRNN(nn.Module):

    def __init__(self, x_dim, h_dim, z_dim, d_dim, n_layers, bias=False):
        super(VRNN, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.d_dim = d_dim

        self.x_0 = Parameter(torch.rand(self.x_dim))
        self.h_0 = Parameter(torch.rand(self.h_dim))

        # recurrence
        self.rnn = nn.LSTM(x_dim + z_dim, h_dim, n_layers, bias)

        # feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU())
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU())

        # encoder
        self.enc = nn.Sequential(
            nn.Linear(h_dim + h_dim + d_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(h_dim + d_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())

    def forward(self, x, d):

        z = []
        recons_loss = 0
        kld_loss = 0

        h = Variable(torch.rand(self.n_layers, x.size(1), self.h_dim))
        # z = Variable(torch.zeros(self.n_layers, x.size(1), self.z_dim))
        for t in range(-1, x.size(0) - 1):

            if t == -1:
                # encoder
                enc_t = self.enc(torch.cat([self.x_0, self.h_0, d], 1))
                enc_mean_t = self.enc_mean(enc_t)
                enc_std_t = self.enc_std(enc_t)
            else:
                phi_x_t = self.phi_x(x[t])
                # recurrence
                _, h = self.rnn(torch.cat([x[t], z_t], 1).unsqueeze(0), h)
                # encoder
                enc_t = self.enc(torch.cat([phi_x_t, h[-1], d], 1))
                enc_mean_t = self.enc_mean(enc_t)
                enc_std_t = self.enc_std(enc_t)

            #sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, d], 1))
            # computing losses
            kld_loss = _compute_kld_loss(enc_std_t, enc_mean_t)
            recons_loss += F.mse_loss(dec_t, x[t + 1])

            z.append(z_t)

        return kld_loss, recons_loss

    # def sample(self, seq_len):

    #     sample = torch.zeros(seq_len, self.x_dim)

    #     h = Variable(torch.zeros(self.n_layers, 1, self.h_dim))
    #     for t in range(seq_len):

    #         # prior
    #         prior_t = self.prior(h[-1])
    #         prior_mean_t = self.prior_mean(prior_t)
    #         prior_std_t = self.prior_std(prior_t)

    #         #sampling and reparameterization
    #         z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
    #         phi_z_t = self.phi_z(z_t)

    #         # decoder
    #         dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
    #         dec_mean_t = self.dec_mean(dec_t)
    #         #dec_std_t = self.dec_std(dec_t)

    #         phi_x_t = self.phi_x(dec_mean_t)

    #         # recurrence
    #         _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

    #         sample[t] = dec_mean_t.data

    #     return sample

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def _init_weights(self, stdv):
        pass

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def _compute_kld_loss(self, log_var, mu):
        return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
