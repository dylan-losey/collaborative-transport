import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle
import random
import copy
import numpy as np
import sys
import os




# RNN-RNN CVAE, based on "learning latent plans from play"
# Here we implement with LSTMs at Encoder and Decoder
class LSTM_LSTM(nn.Module):


    def __init__(self):
        super(LSTM_LSTM, self).__init__()

        # encoder
        self.hidden_size_enc = 3
        self.input_size_enc = 8
        self.output_size_enc = 1
        self.lstm_enc = nn.LSTM(self.input_size_enc, self.hidden_size_enc)
        self.fc_enc_1 = nn.Linear(self.hidden_size_enc, 2*self.hidden_size_enc)
        self.fc_enc_2 = nn.Linear(2*self.hidden_size_enc, self.output_size_enc)

        # decoder
        self.hidden_size_dec = 10
        self.input_size_dec = 7
        self.output_size_dec = 2
        self.lstm = nn.LSTM(self.input_size_dec, self.hidden_size_dec)
        self.fc_1 = nn.Linear(self.hidden_size_dec, 2*self.hidden_size_dec)
        self.fc_2 = nn.Linear(2*self.hidden_size_dec, 2*self.hidden_size_dec)
        self.fc_3 = nn.Linear(2*self.hidden_size_dec, self.output_size_dec)

        # loss function
        self.relu = nn.ReLU()
        self.loss = nn.MSELoss()


    # encode trajectory to latent state z
    def encode(self, x):
        hidden = (torch.zeros(1,1,self.hidden_size_enc), torch.zeros(1,1,self.hidden_size_enc))
        for count, input in enumerate(x):
            output, hidden = self.lstm_enc(input.view(1,1,-1), hidden)
        h1 = self.relu(self.fc_enc_1(output[0,0,:]))
        return self.fc_enc_2(h1)


    # decode each input to a human action
    def decode(self, s, z):
        a_hat = torch.zeros(len(s), 2)
        hidden = (torch.zeros(1,1,self.hidden_size_dec), torch.zeros(1,1,self.hidden_size_dec))
        for count, input in enumerate(s):
            input_with_z = torch.cat((input, z), 0)
            output, hidden = self.lstm(input_with_z.view(1,1,-1), hidden)
            h1 = self.relu(self.fc_1(output[0,0,:]))
            h2 = self.relu(self.fc_2(h1))
            a_hat[count,:] = self.fc_3(h2)
        return a_hat


    # initialize hidden
    def init_hidden(self):
        return (torch.zeros(1,1,self.hidden_size_dec), torch.zeros(1,1,self.hidden_size_dec))


    # decode a step
    def step(self, s, z, hidden):
        input_with_z = torch.cat((s, z), 0)
        output, hidden = self.lstm(input_with_z.view(1,1,-1), hidden)
        h1 = self.relu(self.fc_1(output[0,0,:]))
        h2 = self.relu(self.fc_2(h1))
        a_hat = self.fc_3(h2)
        return a_hat, hidden


    def forward(self, x, s):
        z = self.encode(x)
        a_hat = self.decode(s, z)
        return a_hat



def main():

    EPOCH = 10000
    LR = 0.01
    LR_STEP_SIZE = 2000
    LR_GAMMA = 0.1
    savename = "models/lstm_model.pt"

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

    model = LSTM_LSTM()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    for epoch in range(EPOCH):
        random.shuffle(data_full)
        data = torch.tensor(data_full)
        optimizer.zero_grad()
        loss = 0.0
        for traj in data:
            x = traj[:,0:8]
            s = traj[:,0:6]
            a = traj[:,6:8]
            a_hat = model(x, s)
            loss += model.loss(a, a_hat)
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(epoch, loss.item())
        torch.save(model.state_dict(), savename)


if __name__ == "__main__":
    main()
