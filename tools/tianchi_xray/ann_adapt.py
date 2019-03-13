import simplejson as json


def main():
    objs = {}
    with open('./datasets/tianchi_xray/train_no_poly.json') as f:
        objs = json.load(f)

    for ann in objs['annotations']:
        ann['area'] = 2

    with open('./datasets/tianchi_xray/train_no_poly_adapted.json', 'w') as f:
        json.dump(objs, f)


if __name__ == '__main__':
    main()
