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



# I had trouble with an LSTM (which was my first choice here)
# The LSTM just learned to take the mean action, even in the simplest case
# Talking with Krishnan, it sounds like the gradients may not have been passed correctly
# With this RNN the robot was able to reasonably overfit to a single trajectory
class RNNAE(nn.Module):

    def __init__(self):
        super(RNNAE, self).__init__()

        self.hidden_size_enc = 3
        self.input_size_enc = 8
        self.output_size_enc = 1

        self.i2h_enc = nn.Linear(self.input_size_enc + self.hidden_size_enc, self.hidden_size_enc)
        self.i2o_enc = nn.Linear(self.input_size_enc + self.hidden_size_enc, self.output_size_enc)

        self.hidden_size = 5
        self.input_size = 7
        self.output_size = 2

        self.i2h = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.i2o_1 = nn.Linear(self.input_size + self.hidden_size, 2*self.input_size)
        self.i2o_2 = nn.Linear(2*self.input_size, 2*self.input_size)
        self.i2o_3 = nn.Linear(2*self.input_size, self.output_size)
        self.relu = nn.ReLU()

        self.loss = nn.MSELoss()


    def encode_step(self, input, hidden):
        combined = torch.cat((input, hidden), 0)
        hidden = self.i2h_enc(combined)
        output = self.i2o_enc(combined)
        return output, hidden


    def decode_step(self, input, hidden):
        combined = torch.cat((input, hidden), 0)
        hidden = self.i2h(combined)
        h1 = self.relu(self.i2o_1(combined))
        h2 = self.relu(self.i2o_2(h1))
        output = self.i2o_3(h2)
        return output, hidden


    def encode(self, x):
        hidden = torch.randn(self.hidden_size_enc)
        for count, input in enumerate(x):
            output, hidden = self.encode_step(input, hidden)
        return output


    def decode(self, s, z):
        a_hat = torch.zeros(len(s), 2)
        hidden = torch.randn(self.hidden_size)
        for count, input in enumerate(s):
            input_with_z = torch.cat((input, z), 0)
            output, hidden = self.decode_step(input_with_z, hidden)
            a_hat[count,:] = output
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
