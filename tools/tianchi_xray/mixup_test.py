import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def main():
    names = list(os.listdir('datasets/tianchi_xray/restricted/'))
    img1 = Image.open(os.path.join('datasets/tianchi_xray/restricted', names[random.randint(0, len(names))]))
    img2 = Image.open(os.path.join('datasets/tianchi_xray/restricted', names[random.randint(0, len(names))]))

    plt.subplot(2, 2, 1)
    plt.imshow(img1)
    plt.subplot(2, 2, 2)
    plt.imshow(img2)

    lambd = 0.5

    img1 = np.array(img1, dtype='float32')
    img2 = np.array(img2, dtype='float32')
    height = max(img1.shape[0], img2.shape[0])
    width = max(img1.shape[1], img2.shape[1])

    mixed_img = np.zeros(shape=(height, width, 3), dtype='float32')
    mixed_img[:img1.shape[0], :img1.shape[1], :] = img1 * lambd
    mixed_img[:img2.shape[0], :img2.shape[1], :] += img2 * (1. - lambd)
    mixed_img = mixed_img.astype('uint8')
    mixed_img = Image.fromarray(mixed_img)

    plt.subplot(2, 2, 3)
    plt.imshow(mixed_img)
    plt.show()


if __name__ == '__main__':
    main()
