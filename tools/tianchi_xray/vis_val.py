import os, sys
import simplejson as json
import torch

this_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(this_dir, '..', '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


def main():
    preds = torch.load('./models/predictions.pth')
    objs = []
    print(len(preds))
    for pred in preds:
        extra = pred.extra_fields
        bbox = []
        for box in pred.bbox:
            bbox.append(list(map(lambda a: float(a), list(box))))
        labels = list(map(lambda x: int(x), extra['labels']))
        scores = list(map(lambda x: float(x), extra['scores']))
        entry = {
            'bbox': bbox,
            'labels': labels,
            'scores': scores,
            'size': pred.size,
            'mode': pred.mode,
        }
        objs.append(entry)

    with open('./models/predictions.json', 'w') as f:
        json.dump(objs, f)


if __name__ == '__main__':
    main()
