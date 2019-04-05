import os
import sys
import simplejson as json

this_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(this_dir, '..', '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import torch

pred_path = 'datasets/tianchi_xray/predictions.pth'


def main():
    boxlists = torch.load(pred_path)
    for boxlist in boxlists:
        print(boxlist)


if __name__ == '__main__':
    main()
