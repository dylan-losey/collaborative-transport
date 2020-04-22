import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
import copy
import numpy as np
import sys
import os



# RNN-RNN CVAE, based on "learning latent plans from play"
# Here we implement with LSTMs at Encoder and Decoder
class RNNAE(nn.Module):


    def __init__(self):
        super(RNNAE, self).__init__()

        # encoder
        self.hidden_size_enc = 3
        self.input_size_enc = 8
        self.output_size_enc = 1
        self.lstm_enc = nn.LSTM(self.input_size_enc, self.hidden_size_enc)
        self.fc_enc = nn.Linear(self.hidden_size_enc, self.output_size_enc)

        # decoder
        self.hidden_size_dec = 5
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
        hidden = (torch.randn(1,1,self.hidden_size_enc), torch.randn(1,1,self.hidden_size_enc))
        for count, input in enumerate(x):
            output, hidden = self.lstm_enc(input.view(1,1,-1), hidden)
        return self.fc_enc(output[0,0,:])


    # decode each input to a human action
    def decode(self, s, z):
        a_hat = torch.zeros(len(s), 2)
        hidden = (torch.randn(1,1,self.hidden_size_dec), torch.randn(1,1,self.hidden_size_dec))
        for count, input in enumerate(s):
            input_with_z = torch.cat((input, z), 0)
            output, hidden = self.lstm(input_with_z.view(1,1,-1), hidden)
            h1 = self.relu(self.fc_1(output[0,0,:]))
            h2 = self.relu(self.fc_2(h1))
            a_hat[count,:] = self.fc_3(h2)
        return a_hat


    def forward(self, x, s):
        z = self.encode(x)
        a_hat = self.decode(s, z)
        return a_hat


EPOCH = 10000
LR = 0.01
LR_STEP_SIZE = 2000
LR_GAMMA = 0.1
DATANAME = "models/traj_dataset.pkl"
SAVENAME = "models/test-lstm.pt"


def main():

    data = pickle.load(open(DATANAME, "rb"))
    data = torch.Tensor(data)
    model = RNNAE()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    for idx in range(EPOCH):

        optimizer.zero_grad()
        loss = 0.0

        for traj in data:
            x = traj[:,1:]
            s = x[:,0:6]
            a = x[:,6:8]
            a_hat = model(x, s)
            loss += model.loss(a, a_hat)

        loss.backward()
        optimizer.step()
        scheduler.step()
        print(idx, loss.item())
        torch.save(model.state_dict(), SAVENAME)


if __name__ == "__main__":
    main()
