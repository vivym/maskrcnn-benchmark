import simplejson as json
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


path_load = 'datasets/tianchi_xray/eval_no_poly_adapted_no_normal_bbox.json'
# path_load = 'datasets/tianchi_xray/train_no_poly_adapted_ex_eval_with_normal.json'
path_save = 'datasets/tianchi_xray/eval_no_poly_adapted_no_normal_bbox_with_area.json'


def calc_area(p1, p2, p3, p4):
    v1 = p2 - p1
    v2 = p3 - p1
    v3 = p4 - p1
    return (abs(v1 * v2) + abs(v2 * v3)) / 2.0


def main():
    with open(path_load) as f:
        dataset = json.load(f)

    anns = dataset['annotations']
    areas = []
    for ann in anns:
        '''
        if 'minAreaRect' not in ann:
            continue
        rect = ann['minAreaRect']
        if len(rect) != 4:
            continue
        p1, p2, p3, p4 = list(map(lambda x: Point(*x), rect))
        area = calc_area(p1, p2, p3, p4)
        '''
        _, _, w, h = ann['bbox']
        assert w * h > 0
        area = w * h
        ann['area'] = area
        areas.append(area)

    # with open(path_save, 'w') as f:
    #     json.dump(dataset, f)
    # plt.hist(areas, bins=50)
    # plt.show()


if __name__ == '__main__':
    main()
