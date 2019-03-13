import simplejson as json
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline


def load(path):
    objs = []
    with open(path) as f:
        for line in f.readlines():
            objs.append(json.loads(line))

    iters = list(map(lambda x: x['iter'], objs))
    aps = list(map(lambda x: x['results']['bbox']['AP'], objs))

    iters = np.asarray(iters)
    aps = np.asarray(aps)

    # print('\n'.join(map(lambda x: str(x), zip(iters, aps))))

    # spline(iters.min(), iters.max(), )

    plt.plot(iters, aps)



def main():
    load('datasets/tianchi_xray/pred/all_eval_no_normal_bbox_mixup.txt')

    plt.show()


if __name__ == '__main__':
    main()
