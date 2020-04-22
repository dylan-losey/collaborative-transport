import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle
import copy
import numpy as np
import sys
import os
from test_rnn import RNNAE



class Model(object):

    def __init__(self, modelname):
        self.model = RNNAE()
        model_dict = torch.load(modelname, map_location='cpu')
        self.model.load_state_dict(model_dict)
        self.model.eval

    def forward(self, x, s):
        return self.model.forward(x, s)

    def encode(self, x):
        return self.model.encode(x)

    def decode(self, s, z):
        return self.model.decode(s, z)


def plot_trust2z(model):

    data = pickle.load(open(DATASET, "rb"))
    data = torch.Tensor(data)

    trust, Z = [], []
    for traj in data:
        x = traj[:,1:]
        z_pred = model.encode(x)
        z_true = traj[0, 0]
        trust.append(z_true.item())
        Z.append(z_pred.item())

    plt.plot(trust, Z, 'bo')
    plt.show()


def plot_action(model):

    data = pickle.load(open(DATASET, "rb"))
    data = torch.Tensor(data)

    for traj in data:
        x = traj[:,1:]
        s = x[:,0:6]
        a = x[:,6:8]
        z = traj[0, 0].item()
        ahat = model.forward(x, s)

        print(round(z*10))
        plt.plot(a.numpy(),'--')
        plt.plot(ahat.detach().numpy(),'o')
        plt.show()


def plot_zrollout(model):

    data = pickle.load(open(DATASET, "rb"))
    data = torch.Tensor(data)

    Z = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for traj in data:
        x = traj[:,1:]
        s = x[:,0:6]
        ahuman = x[:,6:8]
        arobot = x[:,4:6]
        for z in Z:
            zt = torch.tensor(z).view(1)
            ahat = model.decode(s, zt)
            plt.plot(ahat.detach().numpy(),'-')
        plt.plot(ahuman.numpy(),'x')
        plt.plot(arobot.numpy(),'s')
        plt.show()


DATASET = "models/traj_dataset.pkl"

def main():

    modelname = 'models/test-lstm.pt'
    model = Model(modelname)

    plot_trust2z(model)
    plot_action(model)
    plot_zrollout(model)


if __name__ == "__main__":
    main()
