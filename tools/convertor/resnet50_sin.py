from collections import OrderedDict

import torch


def main():
    objs = torch.load(
        './models/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
        # './models/resnet50-19c8e357.pth',
        map_location='cpu'
    )

    state_dict = objs
    if 'state_dict' in objs:
        state_dict = state_dict['state_dict']

    new_dict = OrderedDict()
    for key in state_dict.keys():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[7:]
        if new_key.startswith("fc"):
            continue
        if new_key.startswith("layer4"):
            continue
        if new_key.startswith("layer3"):
            continue
        if new_key.startswith("layer2"):
            continue
        if new_key.startswith("layer1.2"):
            continue
        if new_key.startswith("layer1.1"):
            continue
        if new_key.startswith("layer1.0"):
            continue
        type = '' if new_key.startswith('layer') else 'stem.'
        new_key = 'backbone.body.' + type + new_key
        print(key, '--->', new_key)
        new_dict[new_key] = state_dict[key]

    torch.save({
      'model': new_dict,
    } ,"./models/resnet50_sin.pth")


if __name__ == '__main__':
    main()
