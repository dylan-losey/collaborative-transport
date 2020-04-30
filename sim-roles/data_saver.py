import pickle
import sys
import os
import numpy as np
import matplotlib.pyplot as plt



def main():

    dataset = []
    folder = 'simulated-data/'
    savename = 'models/dataset.pkl'

    for filename in os.listdir(folder):
        local_data = pickle.load(open(folder + filename, "rb"))
        traj = []
        for count, item in enumerate(local_data):
            item_x = item[2:6] + item[8:12]
            item_x[0] = (item_x[0] - 400) / 350.0
            item_x[1] = (item_x[1] - 400) / 350.0
            item_x[2] = (item_x[2] - 400) / 350.0
            item_x[3] = (item_x[3] - 400) / 350.0
            item_x[4] = item_x[4] / 50.0
            item_x[5] = item_x[5] / 50.0
            item_x[6] = item_x[6] / 50.0
            item_x[7] = item_x[7] / 50.0
            if count > 15 and count < 21:
                traj.append([item[0], item[1]] + item_x)

        print(filename, len(traj))
        dataset.append(traj)

    pickle.dump(dataset, open(savename, "wb"))
    print(len(dataset))


if __name__ == "__main__":
    main()
