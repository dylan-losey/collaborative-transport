import pickle
import sys
import os
import numpy as np
import matplotlib.pyplot as plt



def main():

    dataset = []
    folder = 'trajectories/r_obs_300/'
    savename = 'datasets/r_obs_300.pkl'

    for filename in os.listdir(folder):
        local_data = pickle.load(open(folder + filename, "rb"))
        follow = float(filename[2:5])
        traj = []
        for count, item in enumerate(local_data):
            if count >= 25:
                break
            x = [0] * 9
            x[0] = (item[0] - 400) / 350.0
            x[1] = (item[1] - 400) / 350.0
            x[2] = (item[2] - 400) / 350.0
            x[3] = (item[3] - 400) / 350.0
            x[4] = item[6] / 50.0
            x[5] = item[7] / 50.0
            x[6] = item[8] / 50.0
            x[7] = item[9] / 50.0
            x[8] = follow
            traj.append(x)

        print(filename, len(traj))
        dataset.append(traj)

    pickle.dump(dataset, open(savename, "wb"))
    print(len(dataset))


if __name__ == "__main__":
    main()
