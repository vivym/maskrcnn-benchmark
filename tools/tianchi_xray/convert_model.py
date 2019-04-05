from collections import OrderedDict

import torch


def main():
    objs = torch.load('./models/e2e_mask_rcnn_X_101_32x8d_FPN_1x_without_box_and_mask.pth', map_location='cpu')
    model = objs['model']
    print(sum((v.numel() for _, v in model.items()), ))

    new_model = OrderedDict()
    for key in model.keys():
        if not key.startswith('module.roi_heads'):
            print(key)
            new_model[key] = model[key]

    torch.save({
        'model': new_model,
    }, './models/e2e_mask_rcnn_X_101_32x8d_FPN_1x_without_roi_head.pth')


if __name__ == '__main__':
    main()
