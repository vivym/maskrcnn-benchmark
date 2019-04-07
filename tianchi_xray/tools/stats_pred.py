import os, sys
import simplejson as json

this_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(this_dir, '..', '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import torch


def main():
    preds = torch.load('datasets/tianchi_xray/pred2/predictions.pth')

    print(len(preds))


if __name__ == '__main__':
    main()
