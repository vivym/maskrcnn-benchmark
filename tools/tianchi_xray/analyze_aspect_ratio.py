import simplejson as json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


def load_dataset(path):
    with open(path) as f:
        dataset = json.load(f)

    return dataset['images'], dataset['annotations'], dataset


def main():
    images, anns, dataset = load_dataset('datasets/tianchi_xray/train_no_poly_adapted.json')

    ratios = []
    for img in images:
        width, height = img['width'], img['height']
        ratios.append(width / height)

    ratios = np.asarray(ratios)
    mu = np.mean(ratios)
    sigma = np.std(ratios)
    print(mu, sigma)
    n, bins, patches = plt.hist(ratios, bins=20, normed=1)
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y)
    plt.show()


if __name__ == '__main__':
    main()
