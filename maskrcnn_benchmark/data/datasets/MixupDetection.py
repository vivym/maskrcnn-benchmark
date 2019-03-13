import torch
import torch.utils.data as data
import numpy as np
from PIL import Image

from maskrcnn_benchmark.structures.bounding_box import BoxList


class MixupDetection(data.Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset.
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Pytorch dataset object.
    mixup : callable random generator, e.g. np.random.uniform
        A random mixup ratio sampler, preferably a random generator from numpy.random
        A random float will be sampled each time with mixup(*args).
        Use None to disable.
    *args : list
        Additional arguments for mixup random sampler.
    """

    def __init__(self, dataset, transform, mixup=None, *args):
        self.dataset = dataset
        self.transform = transform
        self.mixup = mixup
        self.mixup_args = args
        self.lambd = 1.0

    def set_mixup(self, mixup=None, *args):
        self.mixup = mixup
        self.mixup_args = args

    def step(self):
        if self.mixup is not None:
            self.lambd = max(0, min(1, self.mixup(*self.mixup_args)))
            return self.lambd
        else:
            return None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img1, target1, _ = self.dataset[idx]

        lambd = self.lambd

        if lambd >= 1:
            if self.transform is not None:
                img1, target1 = self.transform(img1, target1)
            return img1, target1, idx

        idx2 = np.random.choice(np.delete(np.arange(len(self)), idx))
        img2, target2, _ = self.dataset[idx2]

        img1 = np.array(img1, dtype='float32')
        img2 = np.array(img2, dtype='float32')
        height = max(img1.shape[0], img2.shape[0])
        width = max(img1.shape[1], img2.shape[1])

        mixed_img = np.zeros(shape=(height, width, 3), dtype='float32')
        mixed_img[:img1.shape[0], :img1.shape[1], :] = img1 * lambd
        mixed_img[:img2.shape[0], :img2.shape[1], :] += img2 * (1. - lambd)
        mixed_img = mixed_img.astype('uint8')
        mixed_img = Image.fromarray(mixed_img)

        assert target1.mode == target2.mode
        mixed_target = BoxList(
            bbox=torch.cat((target1.bbox, target2.bbox)).to(target1.bbox.device),
            image_size=(width, height),
            mode=target1.mode
        )
        mixed_target.extra_fields['labels'] = torch.cat((
            target1.extra_fields['labels'],
            target2.extra_fields['labels'],
        )).to(target1.extra_fields['labels'].device)

        if self.transform is not None:
            mixed_img, mixed_target = self.transform(mixed_img, mixed_target)

        return mixed_img, mixed_target, idx, lambd

    def get_img_info(self, index):
        return self.dataset.get_img_info(index)
