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
from lstm_role import RNNAE



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

    Z = np.zeros((35 - TRAJ_LEN, len(data)))
    for idx, traj in enumerate(data):
        traj = torch.Tensor(traj)
        for snip in range(35 - TRAJ_LEN):
            x = traj[snip:snip+TRAJ_LEN,1:]
            z_pred = model.encode(x)
            Z[snip, idx] = z_pred.item()

    plt.plot(Z)
    plt.show()


def plot_action(model):

    data = pickle.load(open(DATASET, "rb"))
    data = torch.Tensor(data)

    for traj in data:
        A = np.zeros((len(traj),2))
        Ahat = np.zeros((len(traj),2))
        for snip in range(len(traj) - TRAJ_LEN):
            x = traj[snip:snip+TRAJ_LEN,1:]
            s = x[:,0:6]
            a = x[:,6:8]
            z = traj[0, 0].item()
            ahat = model.forward(x, s)
            A[snip:snip+TRAJ_LEN,:] = a.numpy()
            Ahat[snip:snip+TRAJ_LEN,:] = ahat.detach().numpy()

        print(round(z*10))
        plt.plot(A,'--')
        plt.plot(Ahat,'o')
        plt.show()


def plot_zrollout(model):

    data = pickle.load(open(DATASET, "rb"))
    data = torch.Tensor(data)

    Z = [-3.0]

    traj = data[3]
    for z in Z:
        Ah = np.zeros((len(traj),2))
        Ar = np.zeros((len(traj),2))
        Ahat = np.zeros((len(traj),2))
        for snip in range(len(traj) - TRAJ_LEN):
            x = traj[snip:snip+TRAJ_LEN,1:]
            s = x[:,0:6]
            ar = x[:,4:6]
            ah = x[:,6:8]
            zt = torch.tensor(z).view(1)
            ahat = model.decode(s, zt)
            Ar[snip:snip+TRAJ_LEN,:] = ar.numpy()
            Ah[snip:snip+TRAJ_LEN,:] = ah.numpy()
            Ahat[snip:snip+TRAJ_LEN,:] = ahat.detach().numpy()

        plt.plot(Ar,'s-')
        # plt.plot(Ah,'o-')
        plt.plot(Ahat,'--')
        plt.show()


DATASET = "models/noise_dataset_2.pkl"
TRAJ_LEN = 2

def main():

    modelname = 'models/lstm_noise.pt'
    model = Model(modelname)

    # plot_trust2z(model)
    # plot_action(model)
    plot_zrollout(model)


if __name__ == "__main__":
    main()
