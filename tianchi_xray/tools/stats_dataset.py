import simplejson as json


def main():
    with open('datasets/tianchi_xray/train_no_poly_adapted_ex_eval_with_normal.json') as f:
        dataset1 = json.load(f)

    with open('datasets/tianchi_xray/train_no_poly_adapted_ex_eval_with_normal2.json') as f:
        dataset2 = json.load(f)

    print(len(dataset1['images']))
    print(len(dataset2['images']))
    for ann1, ann2 in zip(dataset1['annotations'], dataset2['annotations']):
        if ann1['image_id'] != ann1['image_id']:
            print(ann1, ann1)


if __name__ == '__main__':
    main()
