import simplejson as json
import os, sys
from PIL import Image
import random


def load_dataset(path):
    dataset = {}
    with open(path) as f:
        dataset = json.load(f)

    return dataset['images'], dataset['annotations'], dataset


def save_dataset(dataset, path):
    with open(path, 'w') as f:
        json.dump(dataset, f)


def filter_bbox(score):
    pass


def main():
    # normal_images, normal_anns = load_dataset('./datatsets/tianchi_xray/train_no_poly_normal_gen.json')
    train_images, train_anns, train_dataset = load_dataset('./datasets/tianchi_xray/train_no_poly_adapted_ex_eval.json')
    eval_images, eval_anns, eval_dataset = load_dataset('./datasets/tianchi_xray/eval_no_poly_adapted.json')

    normal_images = []
    for id, name in enumerate(os.listdir('./datasets/tianchi_xray/normal')):
        img = Image.open(os.path.join('datasets', 'tianchi_xray', 'normal', name))
        width, height = img.size
        name = os.path.join('..', 'normal', name)
        entry = {
            "coco_url": "",
            "data_captured": "",
            "file_name": name,
            "flickr_url": "",
            "id": id + 100000,
            "height": height,
            "width": width,
            "license": 1,
        }
        normal_images.append(entry)

    print(len(normal_images))
    random.shuffle(normal_images)

    eval_images.extend(normal_images[:400])
    eval_dataset['images'] = eval_images

    train_images.extend(normal_images[400:])
    train_dataset['images'] = train_images

    print('train\t', len(train_images))
    print('test\t', len(eval_images))

    save_dataset(train_dataset, './datasets/tianchi_xray/train_no_poly_adapted_ex_eval_with_normal.json')
    save_dataset(eval_dataset, './datasets/tianchi_xray/eval_no_poly_adapted_with_normal.json')


def fix():
    images = {}
    with open('./datasets/tianchi_xray/eval_no_poly_adapted_with_normal.json') as f:
        images = json.load(f)

    eval_images, eval_anns, eval_dataset = load_dataset('./datasets/tianchi_xray/eval_no_poly_adapted.json')

    eval_dataset['images'] = images

    save_dataset(eval_dataset, './datasets/tianchi_xray/eval_no_poly_adapted_with_normal.json')


if __name__ == '__main__':
    main()
