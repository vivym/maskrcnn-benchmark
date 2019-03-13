import os
import sys
import argparse
import simplejson as json
import torch

this_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(this_dir, '..', '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


def load_pred(path):
    boxlists = torch.load(path)
    return boxlists

    for boxlist in boxlists:
        print(boxlist.bbox)
        print(boxlist.extra_fields)
        entry = {
            'filename': '',

        }
        return


def load_images(path):
    objs = {}
    with open(path) as f:
        objs = json.load(f)

    return objs['images']


def main():
    preds = load_pred('./datasets/tianchi_xray/predictions.pth')
    print(len(preds))

    images = load_images('./datasets/tianchi_xray/test_no_poly_a.json')

    results = []
    for idx, image in enumerate(images):
        entry = {
            'filename': image['file_name'],
            'rects': []
        }
        width = image['width']
        height = image['height']
        pred = preds[idx]
        pred = pred.convert('xyxy')
        x_scale = width / pred.size[0]
        y_scale = height / pred.size[1]
        rects = []
        bboxes = pred.bbox.tolist()
        labels = pred.extra_fields['labels'].tolist()
        scores = pred.extra_fields['scores'].tolist()
        for id, bbox in enumerate(bboxes):
            if labels[id] not in range(1, 6) or scores[id] < 0.05:
                continue
            rects.append({
                'xmin': bbox[0] * x_scale,
                'xmax': bbox[2] * x_scale,
                'ymin': bbox[1] * y_scale,
                'ymax': bbox[3] * y_scale,
                'label': labels[id],
                'confidence': scores[id],
            })

        entry['rects'] = rects
        results.append(entry)

    with open('./datasets/tianchi_xray/submit.json', 'w') as f:
        json.dump({
            'results': results
        }, f)


if __name__ == '__main__':
    main()
