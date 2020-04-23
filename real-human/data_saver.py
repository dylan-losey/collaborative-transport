import pickle
import sys
import os
import numpy as np
import matplotlib.pyplot as plt



def main():

    dataset = []
    folder = 'experimental-data'
    savename = 'dataset/dylan_test.pkl'

    n_traj = len([name for name in os.listdir(folder)])
    for run in range(n_traj):
        filename = "r" + str(run) + ".pkl"
        traj = pickle.load(open(folder + "/" + filename, "rb"))
        traj_normalized = []
        for count, item in enumerate(traj):
            item[0] = (item[0] - 400) / 350.0
            item[1] = (item[1] - 400) / 350.0
            item[2] = (item[2] - 400) / 350.0
            item[3] = (item[3] - 400) / 350.0
            item[4] = item[4] / 50.0
            item[5] = item[5] / 50.0
            item[6] = item[6] / 50.0
            item[7] = item[7] / 50.0
            item[8] = (item[8] - 400) / 350.0
            item[9] = (item[9] - 400) / 350.0
            traj_normalized.append(item)
        traj_subsampled = []
        count = 0
        for idx in range(0,len(traj),10):
            count += 1
            if count > 18:
                break
            traj_subsampled.append(traj_normalized[idx])

        print(filename, len(traj_subsampled))

        # T = np.asarray(traj_subsampled)
        # plt.plot(T)
        # plt.show()

        dataset.append(traj_subsampled)

    pickle.dump(dataset, open(savename, "wb"))
    print(len(dataset))


if __name__ == "__main__":
    main()
