import math
import torch
import torch.nn as nn
import torch.utils
import pickle
from pathlib import Path
from torch.autograd import Variable

from model import FLMD

"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""


def pre_train(epoch):
    train_loss = 0
    for batch_idx, (x, d) in enumerate(train_loader):

        #forward + backward + optimize
        optimizer.zero_grad()
        kld_loss, recons_loss, _ = model(x, d)
        loss = kld_loss + nll_loss
        loss.backward()
        optimizer.step()

        # #grad norm clipping, only in pytorch version >= 1.10
        # nn.utils.clip_grad_norm(model.parameters(), clip)

        # printing
        if batch_idx % print_every == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.6f} \t NLL Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                kld_loss.data[0] / batch_size,
                nll_loss.data[0] / batch_size))


# hyperparameters
x_dim = 29
h_dim = 32
z_dim = 16
d_dim = 2
n_layers = 1
n_epochs = 163


learning_rate = 1e-5
decay = 1e-4
batch_size = 512
seed = 2023


# manual seed
torch.manual_seed(seed)
plt.ion()


DATASET = 'Knee'
X = pickle.load(open(Path(DATASET, 'pre_features.pkl'), "rb"))
D = pickle.load(open(Path(DATASET, 'demo_features.pkl'), "rb"))
X = torch.Tensor(X)
D = torch.Tensor(D)

Dataset = TensorDataset(X, D)  # create your datset
train_loader = torch.utils.data.DataLoader(Dataset, batch_size=batch_size, shuffle=True)
save_loader = torch.utils.data.DataLoader(Dataset, batch_size=100000000, shuffle=False)


model = FLMD(x_dim, h_dim, z_dim, d_dim, n_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)


train(epoch)
for batch_idx, (x, d) in enumerate(save_loader):

    _, _, z = model(x, d)
    pickle.dump(z.detach().numpy(), open(Path(DATASET, PRETRAIN + '.pkl'), "wb"))
