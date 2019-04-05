from collections import OrderedDict

import torch

keys_map = {
    'RCNN_layer0.0.weight': 'backbone.body.stem.conv1.weight',
}


def main():
    objs = torch.load('./models/fpn_1_11_58632.pth', map_location='cpu')
    print(objs.keys())
    print(objs['pooling_mode'])
    model = objs['model']
    print(sum((v.numel() for _, v in model.items()), ))

    new_model = OrderedDict()
    for key in model.keys():
        param = model[key]
        if key in keys_map:
            new_model[keys_map[key]] = param
        elif key.startswith('RCNN_layer0.1.'):
            offset = len('RCNN_layer0.1.')
            new_model['backbone.body.stem.bn1.' + key[offset:]] = param
        elif key.startswith('RCNN_layer1'):
            offset = len('RCNN_layer1.0.')
            new_model['backbone.body.layer1.' + key[offset:]] = param
        elif key.startswith('RCNN_layer2'):
            offset = len('RCNN_layer2.0.')
            new_model['backbone.body.layer2.' + key[offset:]] = param
        elif key.startswith('RCNN_layer3'):
            offset = len('RCNN_layer3.0.')
            new_model['backbone.body.layer3.' + key[offset:]] = param
        elif key.startswith('RCNN_layer4'):
            offset = len('RCNN_layer4.0.')
            new_model['backbone.body.layer4.' + key[offset:]] = param
        elif key.startswith('RCNN_layer5'):
            offset = len('RCNN_layer5.0.')
            new_model['backbone.body.layer5.' + key[offset:]] = param
        elif key.startswith('RCNN_toplayer.'):
            offset = len('RCNN_toplayer.')
            new_model['backbone.fpn.fpn_inner5.' + key[offset:]] = param
        elif key.startswith('RCNN_latlayer1.'):
            offset = len('RCNN_latlayer1.')
            new_model['backbone.fpn.fpn_inner4.' + key[offset:]] = param
        elif key.startswith('RCNN_latlayer2.'):
            offset = len('RCNN_latlayer2.')
            new_model['backbone.fpn.fpn_inner3.' + key[offset:]] = param
        elif key.startswith('RCNN_latlayer3.'):
            offset = len('RCNN_latlayer3.')
            new_model['backbone.fpn.fpn_inner2.' + key[offset:]] = param
        elif key.startswith('RCNN_latlayer4.'):
            offset = len('RCNN_latlayer4.')
            new_model['backbone.fpn.fpn_inner1.' + key[offset:]] = param
        elif key.startswith('RCNN_smooth1.'):
            offset = len('RCNN_smooth1.')
            new_model['backbone.fpn.fpn_layer2.' + key[offset:]] = param
        elif key.startswith('RCNN_smooth2.'):
            offset = len('RCNN_smooth2.')
            new_model['backbone.fpn.fpn_layer1.' + key[offset:]] = param
        elif key.startswith('RCNN_rpn.RPN_Conv.'):
            offset = len('RCNN_rpn.RPN_Conv.')
            new_model['rpn.head.conv.' + key[offset:]] = param
        elif key.startswith('RCNN_rpn.RPN_cls_score.'):
            offset = len('RCNN_rpn.RPN_cls_score.')
            # new_model['rpn.head.cls_logits.' + key[offset:]] = param
        elif key.startswith('RCNN_rpn.RPN_bbox_pred.'):
            offset = len('RCNN_rpn.RPN_bbox_pred.')
            # new_model['rpn.head.bbox_pred.' + key[offset:]] = param
        elif key.startswith('RCNN_top.0.'):
            offset = len('RCNN_top.0.')
            new_model['roi_heads.box.feature_extractor.fc6.' + key[offset:]] = param
        elif key.startswith('RCNN_top.2.'):
            offset = len('RCNN_top.2.')
            new_model['roi_heads.box.feature_extractor.fc7.' + key[offset:]] = param
        else:
            print(key)
            print(param.size())

    torch.save({
        'model': new_model,
    }, './models/detnet_with_fpn_and_rpn_head.pth')


if __name__ == '__main__':
    main()
