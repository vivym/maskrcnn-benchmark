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


def main():
    normal_images, normal_anns, _ = load_dataset('./datasets/tianchi_xray/train_no_poly_normal_gen.json')
    train_images, train_anns, train_dataset = load_dataset('./datasets/tianchi_xray/train_no_poly_adapted_ex_eval.json')
    eval_images, eval_anns, eval_dataset = load_dataset('./datasets/tianchi_xray/eval_no_poly_adapted.json')

    normal_obj_categories = [{
        "id": i,
        "name": "normal_obj" + str(i),
        "supercategory": "normal_obj" + str(i),
    } for i in range(6, 11)]

    train_dataset['categories'].extend(normal_obj_categories)
    eval_dataset['categories'].extend(normal_obj_categories)

    id_2_image = {}
    img_id_2_ann = {}
    for entry in normal_images:
        entry['id'] = entry['id'] + 100000
        entry['file_name'] = os.path.join('..', 'normal', entry['file_name'])
        id_2_image[entry['id']] = entry

    normal_anns = list(filter(lambda x: x['score'] > 0.8, normal_anns))
    for entry in normal_anns:
        entry['id'] = entry['id'] + 100000
        entry['image_id'] = entry['image_id'] + 100000
        entry['category_id'] = 6
        img_id_2_ann[entry['image_id']] = entry

    print(len(img_id_2_ann))
    print(len(train_images))
    img_ids = list(img_id_2_ann.keys())

    random.shuffle(img_ids)

    ex_eval_ids = img_ids[500:]
    ex_eval_imgs = list(map(lambda id: id_2_image[id], ex_eval_ids))
    eval_images.extend(ex_eval_imgs)
    eval_dataset['images'] = eval_images
    # eval_anns.extend(normal_anns)
    eval_dataset['annotations'] = eval_anns

    ex_train_ids = img_ids[:500]
    ex_train_imgs = list(map(lambda id: id_2_image[id], ex_train_ids))
    train_images.extend(ex_train_imgs)
    train_dataset['images'] = train_images
    train_anns.extend(normal_anns)
    train_dataset['annotations'] = train_anns

    print('train\t', len(train_images))
    print('test\t', len(eval_images))

    save_dataset(train_dataset, './datasets/tianchi_xray/train_no_poly_adapted_ex_eval_with_normal2.json')
    save_dataset(eval_dataset, './datasets/tianchi_xray/eval_no_poly_adapted_with_normal2.json')


if __name__ == '__main__':
    main()
