import os, sys

this_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(this_dir, '..', '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.backbone.detnet import DetNet

if __name__ == '__main__':
    cfg.MODEL.BACKBONE.CONV_BODY = "D-59-C6"
    cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
    cfg.freeze()
    model = DetNet(cfg)
    print(model)
