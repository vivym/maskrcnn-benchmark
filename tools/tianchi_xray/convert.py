from collections import OrderedDict

import torch


def main():
    objs = torch.load('./models/mask_rcnn_X_101_32x8d_FPN_deform_mixup_0060000_without_box_pred.pth', map_location='cpu')
    model = objs['model']
    print(sum((v.numel() for _, v in model.items()), ))

    new_model = OrderedDict()
    for key in model.keys():
        if not key.startswith('roi_heads.box.feature_extractor.pooler'):
            new_model[key] = model[key]

    torch.save({
        'model': new_model,
    }, './models/mask_rcnn_X_101_32x8d_FPN_deform_mixup_0060000_without_box_pred_and_pooler.pth')


if __name__ == '__main__':
    main()
