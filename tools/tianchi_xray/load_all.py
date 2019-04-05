import simplejson as json
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline


# id_2_label = ['total', '铁壳', '黑钉', '刀具', '电池', '剪刀']

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load(path):
    objs = []
    with open(path) as f:
        for line in f.readlines():
            objs.append(json.loads(line))

    iters = list(map(lambda x: int(x['iter']), objs))
    for i in range(6):
        aps = list(map(lambda x: x['results']['bbox']['AP'][i], objs))
        print('\n'.join(map(lambda x: str(x), enumerate(zip(iters, aps)))))

        iters = np.asarray(iters)
        aps = np.asarray(aps)

        id = aps.argmax()
        # print(iters[id], aps[id])

        # ids = aps > 0.4945
        # print('\n'.join(map(lambda x: str(x), enumerate(filter(lambda x: x[1] < 0.4955, zip(iters[ids], aps[ids]))))))

        # spline(iters.min(), iters.max(), )

        plt.plot(iters, aps, label=str(i))
    plt.legend()


def main():
    load('datasets/tianchi_xray/pred/all.txt')

    plt.show()


if __name__ == '__main__':
    main()
