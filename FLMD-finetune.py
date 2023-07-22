
import math
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn import metrics
from torch.nn.parameter import Parameter

SEED = 658
# np.random.seed(SEED)
# torch.manual_seed(SEED)


# device = torch.device("cpu")

DATASET = 'Knee'

def load_data(DATASET, PRETRAIN, in_l=1):
    pre_features = pickle.load(open(Path(DATASET, 'pre_features.pkl'), "rb"))
    demo_features = pickle.load(open(Path(DATASET, 'demo_features.pkl'), "rb"))
    pretrain_features = pickle.load(open(Path(DATASET, PRETRAIN + '.pkl'), "rb"))
    labels = pickle.load(open(Path(DATASET, 'labels.pkl'), "rb"))  # 40 22

    X = pre_features.values
    y = labels.values[:, in_l]  # 0,1,2
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # X = scaler.fit_transform(X)
    print(X.shape, pretrain_features.shape)
    # X = np.concatenate((X, pretrain_features), axis=-1)
    return X, demo_features.values, pretrain_features, y


class AttentiveRegressor(nn.Module):

    def __init__(self, x_size, d_size, z_size, embed_dim=256):
        super().__init__()

        # self.x_size = x_size
        # self.d_size = d_size
        # self.z_size = z_size

        self.trans_x = nn.Linear(x_size, embed_dim)
        self.trans_d = nn.Linear(d_size, embed_dim)
        self.trans_z = nn.Linear(z_size, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4)
       

    def forward(self, x, d, z):

        return nn.Softmax(self.attention(self.trans_x(x),self.trans_d(d),self.trans_z(z)))


def train_n_evaluate(X, D, Z, y):
    X_train, X_test, p_train, p_test, y_train, y_test = train_test_split(X, Z, y, test_size=0.2)
    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], 1))

    tensor_X_train = torch.Tensor(X_train)  # transform to torch tensor
    tensor_y_train = torch.Tensor(y_train)

    tensor_X_test = torch.Tensor(X_test)  # transform to torch tensor
    tensor_y_test = torch.Tensor(y_test)

    tensor_p_train = torch.Tensor(p_train)  # transform to torch tensor
    tensor_p_test = torch.Tensor(p_test)

    my_dataset = TensorDataset(tensor_X_train, tensor_p_train, tensor_y_train)  # create your datset
    dataloader = DataLoader(my_dataset, batch_size=128)  # create your dataloader

    test_dataset = TensorDataset(tensor_X_test, tensor_p_test, tensor_y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=200000)

    model = nn.AttentiveRegressor(X.shape[1], Z.shape[1])
    # model = Regressor(X.shape[1], p_features.shape[1])
    criterion = nn.CrossEntropyLoss()

    lr = parameter_dict[DATASET + PRETRAIN][0]
    wd = parameter_dict[DATASET + PRETRAIN][1]
    epochs = parameter_dict[DATASET + PRETRAIN][2]
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    LENGTH = len(dataloader)

    for epoch in range(epochs):
        training_loss, training_AUC, test_loss, test_AUC = [], [], 0, 0
        for idx, (x, p, y) in enumerate(dataloader):
            # print(type(labels))
            optimizer.zero_grad()
            output = model(x, p)

            class_sample_count = np.array([len(np.where(y == t)[0]) for t in np.unique(y)])
            weight = 1 / class_sample_count
            samples_weight = torch.Tensor(np.array([weight[int(t)] for t in y]))
            # print(samples_weight)
            loss = criterion(output, y)
            loss = torch.mean(loss * samples_weight)
            loss.backward()
            optimizer.step()

            if idx % 500 == 0:
                torch.no_grad()
                training_loss.append(loss.item())
                np_output = output.detach().numpy()
                np_labels = y.detach().numpy()
                fpr, tpr, thresholds = metrics.roc_curve(np_labels, np_output)
                training_AUC.append(metrics.auc(fpr, tpr))

        for idx, (x, p, y) in enumerate(test_dataloader):
            torch.no_grad()
            output = model(x, p)
            loss = criterion(output, y)
            np_output = output.detach().numpy()
            np_labels = y.detach().numpy()
            fpr, tpr, thresholds = metrics.roc_curve(np_labels, np_output)
            test_AUC = metrics.auc(fpr, tpr)
            test_loss = torch.mean(loss)

        print(f"Training loss: {(np.mean(training_loss)):>0.4f}, \
        Training AUC: {(np.mean(training_AUC)):>0.4f}, \
        Test loss: {test_loss:>4f}, \
        Test AUC: {test_AUC:>4f} ")

    return test_AUC

X, D, Z, y = load_data(DATASET, PRETRAIN, in_l=0)
train_n_evaluate(X, D, Z, y)
