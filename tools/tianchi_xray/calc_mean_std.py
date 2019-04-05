import os
import numpy as np
from PIL import Image
from tqdm import tqdm


# 0.83102435 0.83786940 0.73139444
# 0.27632148 0.20124162 0.29032342
def calc(*paths):
    imgs = []
    for path in paths:
        for name in tqdm(os.listdir(path)):
            img = Image.open(os.path.join(path, name)).convert('RGB')
            img = np.array(img, dtype='float32') # H, W, C
            img = img / 255.0
            img = img.reshape(-1, 3)
            imgs.append(img)

    imgs = np.concatenate(imgs, axis=0)
    # print(imgs.shape)
    # print(imgs.mean(axis=0, dtype=np.float64))
    print(imgs.std(axis=0, dtype=np.float64))


def main():
    calc('datasets/tianchi_xray/restricted', 'datasets/tianchi_xray/normal')


if __name__ == '__main__':
    main()
