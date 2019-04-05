from collections import OrderedDict

import torch


def main():
    objs = torch.load('../../../data/e2e_mask_rcnn_X_101_32x8d_FPN_1x.pth', map_location='cpu')
    print(objs.keys())

    model = objs['model']
    print(sum((v.numel() for _, v in model.items()), ))

    new_model = OrderedDict()
    for key in model.keys():
        if key.startswith('module.roi_heads.box.predictor'):
            continue
        if key.startswith('module.roi_heads.mask'):
            continue

        new_model[key] = model[key]

    print(new_model.keys())

    torch.save({
        'model': new_model,
    }, './models/e2e_mask_rcnn_X_101_32x8d_FPN_1x_without_box_and_mask.pth')


if __name__ == '__main__':
    main()
