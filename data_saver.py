import pickle
import sys
import os
import numpy as np
import matplotlib.pyplot as plt



def main():

    dataset = []
    folder = 'simulated-data'
    savename = 'models/traj_dataset.pkl'

    for filename in os.listdir(folder):
        local_data = pickle.load(open(folder + "/" + filename, "rb"))
        traj = []
        for count, item in enumerate(local_data):
            item_x = item[1:5] + item[7:11]
            if count < 20:
                traj.append([float(filename[1:4])] + item_x)

        print(filename, len(traj))
        dataset.append(traj)

    pickle.dump(dataset, open(savename, "wb"))
    print(len(dataset))

if __name__ == "__main__":
    main()
