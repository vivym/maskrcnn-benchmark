# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T


def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        hflip_prob = cfg.INPUT.HFLIP_PROB_TRAIN
        vflip_prob = cfg.INPUT.VFLIP_PROB_TRAIN
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        hflip_prob = cfg.INPUT.HFLIP_PROB_TEST
        vflip_prob = cfg.INPUT.VFLIP_PROB_TEST

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )

    transform = T.Compose(
        [
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(hflip_prob),
            T.RandomVerticalFlip(vflip_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform
