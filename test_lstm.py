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

        self.input_size = 7
        self.output_size = 2

        self.fc1 = nn.Linear(self.input_size, 2*self.input_size)
        self.fc2 = nn.Linear(2*self.input_size, 2*self.input_size)
        self.fc3 = nn.Linear(2*self.input_size, self.output_size)
        self.relu = nn.ReLU()

        self.loss = nn.MSELoss()


    def encode_step(self, input, hidden):
        combined = torch.cat((input, hidden), 0)
        hidden = self.i2h_enc(combined)
        output = self.i2o_enc(combined)
        return output, hidden


    def decode_step(self, input):
        h1 = self.relu(self.fc1(input))
        h2 = self.relu(self.fc2(h1))
        return self.fc3(h2)


    def encode(self, x):
        hidden = torch.randn(self.hidden_size_enc)
        for count, input in enumerate(x):
            output, hidden = self.encode_step(input, hidden)
        return output


    def decode(self, s, z):
        a_hat = torch.zeros(len(s), 2)
        for count, input in enumerate(s):
            input_with_z = torch.cat((input, z), 0)
            a_hat[count,:] = self.decode_step(input_with_z)
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


def main():

    data = pickle.load(open(DATANAME, "rb"))
    data = torch.Tensor(data)
    # subdata = torch.tensor([data[2], data[5], data[8]])
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
        torch.save(model.state_dict(), "models/test-lstm.pt")
        if loss.item() < 10:
            break



if __name__ == "__main__":
    main()
