import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys
from model import LSTM_LSTM
import random



class Model(object):

    def __init__(self, modelname):
        self.model = LSTM_LSTM()
        model_dict = torch.load(modelname, map_location='cpu')
        self.model.load_state_dict(model_dict)
        self.model.eval

    def forward(self, x, s):
        return self.model.forward(x, s)

    def encode(self, x):
        return self.model.encode(x)

    def decode(self, s, z):
        return self.model.decode(s, z)



def plot_action(data, model):

    for traj in data:
        x = traj[:,0:8]
        s = traj[:,:6]
        ar = traj[:,4:6]
        a = traj[:,6:8]
        ahat = model.forward(x, s)

        plt.plot(a.numpy(),'--')
        plt.plot(ar.numpy(),'-')
        plt.plot(ahat.detach().numpy(),'o')
        plt.show()


def plot_latent(data, model):

    Z, Zstar = [], []
    for traj in data:
        x = traj[:,0:8]
        z_pred = model.encode(x)
        Z.append(z_pred.item())
        Zstar.append(traj[0,8])

    plt.plot(Zstar, Z, 'bo')
    plt.show()


def plot_zrollout(data, model):

    Z = [0.0, 0.5, 1.0]
    for traj in data:
        x = traj[:,0:8]
        s = traj[:,:6]
        ar = traj[:,4:6]
        a = traj[:,6:8]
        for z in Z:
            zt = torch.tensor(z).view(1)
            ahat = model.decode(s, zt)
            plt.plot(ahat.detach().numpy(),'o')
        plt.plot(a.numpy(),'--')
        plt.plot(ar.numpy(),'-')
        plt.show()




def main():

    dataname1 = "datasets/r_obs_200.pkl"
    dataname2 = "datasets/r_obs_225.pkl"
    dataname3 = "datasets/r_obs_250.pkl"
    dataname4 = "datasets/r_obs_275.pkl"
    dataname5 = "datasets/r_obs_300.pkl"
    data1 = pickle.load(open(dataname1, "rb"))
    data2 = pickle.load(open(dataname2, "rb"))
    data3 = pickle.load(open(dataname3, "rb"))
    data4 = pickle.load(open(dataname4, "rb"))
    data5 = pickle.load(open(dataname5, "rb"))
    data_full = data1 + data2 + data3 + data4 + data5
    random.shuffle(data_full)
    data = torch.tensor(data_full)

    modelname = 'models/lstm_model.pt'
    model = Model(modelname)

    # data = data[0:10,:]

    # plot_action(data, model)
    plot_latent(data, model)
    # plot_zrollout(data, model)


if __name__ == "__main__":
    main()
