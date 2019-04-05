import argparse
import os, sys

import cv2
import torch
from torchvision import transforms as T

this_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(this_dir, '..', '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list


def build_transform():
    if cfg.INPUT.TO_BGR255:
        to_bgr_transform = T.Lambda(lambda x: x * 255)
    else:
        to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
    )

    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(min_size, max_size),
            T.ToTensor(),
            to_bgr_transform,
            normalize_transform,
        ]
    )
    return transform


def inference(model, transform, image, device):
    image = transform(image)
    image_list = to_image_list(image, 32)
    image_list = image_list.to(device)


def run(cfg, image_path):
    model = build_detection_model(cfg)
    model.eval()
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    save_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=save_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    transforms = build_transform(cfg)

    image = cv2.imread(image_path)

    prediction = inference(model, transforms, image, device)


def main():
    parser = argparse.ArgumentParser(description='vis')
    parser.add_argument(
        "--config-file",
        default="configs/tianchi_xray/mask_rcnn_X_101_32x8d_FPN_deform_mixup.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--image",
        default="1.jpg",
        metavar="FILE",
        help="path to image file",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    run(cfg, args.image)


if __name__ == '__main__':
    main()
