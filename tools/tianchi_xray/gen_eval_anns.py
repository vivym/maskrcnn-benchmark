import simplejson as json


def load_dataset(path):
    dataset = {}
    with open(path) as f:
        dataset = json.load(f)

    return dataset['images'], dataset['annotations'], dataset


def save_dataset(dataset, path):
    with open(path, 'w') as f:
        json.dump(dataset, f)


def main():
    images, anns, dataset = load_dataset('datasets/tianchi_xray/eval_no_poly_adapted_with_normal.json')
    print(len(anns))
    anns = list(filter(lambda x: x['image_id'] < 100000, anns))
    print(len(anns))

    dataset['annotations'] = anns

    save_dataset(dataset, 'datasets/tianchi_xray/eval_no_poly_adapted_no_normal_bbox.json')


if __name__ == '__main__':
    main()
