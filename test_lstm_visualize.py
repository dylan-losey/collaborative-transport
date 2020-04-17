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
from test_lstm import RNNAE



class Model(object):

    def __init__(self):
        self.model = RNNAE()
        model_dict = torch.load('models/test-lstm.pt', map_location='cpu')
        self.model.load_state_dict(model_dict)
        self.model.eval

    def forward(self, x, s):
        return self.model.forward(x, s)

    def encode(self, x):
        return self.model.encode(x)

    def decode(self, s, z):
        return self.model.decode(s, z)


def main():

    DATANAME = "models/traj_dataset.pkl"
    data = pickle.load(open(DATANAME, "rb"))
    model = Model()

    traj = torch.tensor(data[2])
    x = traj[:,1:]
    print(model.encode(x))

    traj = torch.tensor(data[5])
    x = traj[:,1:]
    print(model.encode(x))

    traj = torch.tensor(data[8])
    x = traj[:,1:]
    print(model.encode(x))

    # traj = torch.tensor(data[2])
    # x = traj[:,1:]
    # s = x[:,0:6]
    # a = x[:,6:8]
    # z = traj[0,0].view(1)
    # traj_hat1 = model.forward(x, s, z)
    #
    # traj_hat2 = model.forward(x, s, torch.Tensor([0.0]).view(1))
    #
    # plt.plot(traj_hat1.detach().numpy(),'-')
    # plt.plot(traj_hat2.detach().numpy(),'-')
    # plt.show()

    for idx in range(11):
        traj = torch.tensor(data[idx])
        x = traj[:,1:]
        s = x[:,0:6]
        arobot = x[:,4:6]
        a = x[:,6:8]
        z = traj[0,0]
        zhat = model.encode(x)
        traj_hat = model.forward(x, s)

        print(z, zhat)
        # plt.plot(arobot.numpy(),'-x')
        plt.plot(a.numpy(),'--o')
        plt.plot(traj_hat.detach().numpy(),'-')
        plt.show()


if __name__ == "__main__":
    main()
