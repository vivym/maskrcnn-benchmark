import simplejson as json


def stats_submit():
    submit = json.load(open('datasets/tianchi_xray/submit_x101_0.539_0.4849.json'))['results']

    total = []
    for ann in submit:
        filename = ann['filename']
        rects = ann['rects']
        if len(rects) <= 0:
            continue
        cnts = [0 for _ in range(5)]
        for rect in rects:
            label = rect['label'] - 1
            cnts[label] = cnts[label] + 1
        mx = max(cnts)
        if mx > 10:
            print(cnts, filename)


def stats_dataset():
    dataset = json.load(open('datasets/tianchi_xray/train_no_poly_adapted.json'))
    images = dataset['images']
    anns = dataset['annotations']

    img_id_2_anns = {}
    for ann in anns:
        img_id = ann['image_id']
        if img_id not in img_id_2_anns:
            img_id_2_anns[img_id] = []
        img_id_2_anns[img_id].append(ann)

    for img in images:
        filename = img['file_name']
        if img['id'] not in img_id_2_anns:
            continue
        anns = img_id_2_anns[img['id']]
        cnts = [0 for _ in range(5)]
        for ann in anns:
            label = ann['category_id'] - 1
            cnts[label] = cnts[label] + 1
        mx = max(cnts)
        if mx > 10:
            print(cnts, filename)


def main():
    stats_dataset()


if __name__ == '__main__':
    main()
