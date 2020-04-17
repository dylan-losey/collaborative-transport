import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import copy


class TrajData(Dataset):

  def __init__(self, filename):
    self.data = pickle.load(open(filename, "rb"))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return torch.FloatTensor(self.data[idx])


class LSTMAE(nn.Module):

    def __init__(self):
        super(LSTMAE, self).__init__()

        self.embed_HD = 3
        self.embed_lstm = nn.LSTM(8, self.embed_HD)
        self.embed_fc = nn.Linear(self.embed_HD, 1)

        self.decode_HD = 128
        self.decode_lstm = nn.LSTM(7, self.decode_HD)
        self.decode_fc = nn.Linear(6, 2)

        self.loss = nn.L1Loss()

    def encoder(self, x):
        hidden = (torch.randn(1, 1, self.embed_HD), torch.randn(1, 1, self.embed_HD))
        for item in x:
            out, hidden = self.embed_lstm(item.view(1, 1, -1), hidden)
        return self.embed_fc(out[0,0,:])

    def decoder(self, s, z):
        hidden = (torch.zeros(1, 1, self.decode_HD), torch.zeros(1, 1, self.decode_HD))
        traj_hat = torch.zeros((len(s), 2))
        for count, item in enumerate(s):
            s_plus_z = torch.cat((item, z), 0)
            out, hidden = self.decode_lstm(s_plus_z.view(1, 1, -1), hidden)
            traj_hat[count,:] = self.decode_fc(item)
        return traj_hat



def main():

    dataname = "models/traj_dataset.pkl"
    data = pickle.load(open(dataname, "rb"))
    traj = torch.tensor(data[70])
    x = traj[:,[1,2,3,4,7,8,9,10]]
    s = x[:,0:6]
    a = x[:,6:8]

    model = LSTMAE()
    EPOCH = 500
    LR = 0.1
    LR_STEP_SIZE = 100
    LR_GAMMA = 0.9

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)


    for idx in range(EPOCH):
        model.zero_grad()
        # z = model.encoder(x)
        z = torch.tensor([1.0])
        traj_hat = model.decoder(s, z)
        loss = model.loss(a, traj_hat)
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(idx, loss.item(), loss)

    print(traj_hat)

if __name__ == "__main__":
    main()
