import simplejson as json
import numpy as np
import matplotlib.pyplot as plt


class Point(object):
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        return self.x * other.y - self.y * other.x

    def __repr__(self):
        return 'Point(x={x}, y={y})'.format(x=self.x, y=self.y)


# path_load = 'datasets/tianchi_xray/eval_no_poly_adapted_no_normal_bbox.json'
path_load = 'datasets/tianchi_xray/train_no_poly_adapted_ex_eval_with_normal.json'
path_save = 'datasets/tianchi_xray/eval_no_poly_adapted_no_normal_bbox_with_area.json'


def calc_area(p1, p2, p3, p4):
    v1 = p2 - p1
    v2 = p3 - p1
    v3 = p4 - p1
    return (abs(v1 * v2) + abs(v2 * v3)) / 2.0


def main():
    with open(path_load) as f:
        dataset = json.load(f)

    images = dataset['images']
    anns = dataset['annotations']

    id_2_images = {}

    for img in images:
        id = img['id']
        id_2_images[id] = img

    areas = {}
    ratios = {}
    cnts = {}
    cnts_areaRng = {}
    imgs_per_cid = {}
    for ann in anns:
        img_id = ann['image_id']
        if img_id not in id_2_images:
            continue

        cid = ann['category_id']
        if cid not in areas:
            areas[cid] = []
            ratios[cid] = []
            cnts[cid] = 0
            cnts_areaRng[cid] = [0, 0, 0]
        cnts[cid] = cnts[cid] + 1

        if cid not in imgs_per_cid:
            imgs_per_cid[cid] = set()
        imgs_per_cid[cid].add(img_id)

        width = id_2_images[img_id]['width']
        height = id_2_images[img_id]['height']
        ratios[cid].append(width / height)

        if 'minAreaRect' not in ann:
            continue

        rect = ann['minAreaRect']
        if len(rect) != 4:
            continue

        p1, p2, p3, p4 = list(map(lambda x: Point(*x), rect))
        area = calc_area(p1, p2, p3, p4)

        _, _, w, h = ann['bbox']
        area = w * h

        areas[cid].append(area)
        rng = 0
        if area > 32 * 32:
            rng = 1
        if area > 96 * 96:
            rng = 2
        cnts_areaRng[cid][rng] = cnts_areaRng[cid][rng] + 1

    for i in range(1, 6):
        print(len(imgs_per_cid[i]))
    print(cnts)
    print(cnts_areaRng)
    # with open(path_save, 'w') as f:
    #     json.dump(dataset, f)
    # plt.hist(areas[5], bins=50)

    idx = 2
    # plt.hist(areas[idx], bins=50)
    area_i = areas[idx]
    area_i.sort()
    area_i_cnt = np.arange(1, len(area_i) + 1)
    # plt.plot(area_i, area_i_cnt)
    plt.hist(ratios[idx], bins=50)
    plt.show()


if __name__ == '__main__':
    main()
