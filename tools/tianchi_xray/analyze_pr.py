import torch
import numpy as np
import matplotlib.pyplot as plt

pr_path = 'datasets/tianchi_xray/tmp.pth'


def main():
    objs = torch.load(pr_path)
    precision = objs['precision']
    recall = objs['recall']
    tp_sum = objs['tp_sum']
    fp_sum = objs['fp_sum']
    pr = objs['pr']
    rc = objs['rc']

    print(precision.shape)  #
    print(recall.shape)     # thresh, category, areaRng, maxDet
    print(len(tp_sum), tp_sum[0].shape) # category, thresh,
    print(len(pr), pr[0].shape)         # category, thresh,

    for i in range(5):
        # plt.plot(np.linspace(0.5, 0.95, 10), recall[:, i, 0, 2], label=str(i))
        plt.plot(np.linspace(0.5, 0.95, 10), rc[i][:,-1], label=str(i))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
