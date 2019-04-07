import os
import simplejson as json
import random
from PIL import Image


dataset_root = 'datasets/round2'

with open(os.path.join(dataset_root, 'train_restriction.json')) as f:
    dataset = json.load(f)

print(len(dataset['annotations']))

images = dataset['images']
anns = dataset['annotations']

img_id_2_anns = {}
img_id_2_img = {}

for img in images:
    img_id_2_img[img['id']] = img

for ann in anns:
    img_id = ann['image_id']
    if img_id not in img_id_2_anns:
        img_id_2_anns[img_id] = []

    img_id_2_anns[img_id].append(ann)

images = []
for img_id in img_id_2_anns.keys():
    images.append(img_id_2_img[img_id])

print(len(img_id_2_anns))

random.seed(1000)
random.shuffle(images)

train_images = images[:1800]
eval_images = images[1800:]

train_ann_count = sum(map(lambda x: len(img_id_2_anns[x['id']]), train_images))
eval_ann_count = sum(map(lambda x: len(img_id_2_anns[x['id']]), eval_images))

print(train_ann_count, eval_ann_count)
print(train_ann_count / len(train_images), eval_ann_count / len(eval_images))

eval_anns = []
for img in eval_images:
    eval_anns.extend(img_id_2_anns[img['id']])

train_anns = []
for img in train_images:
    train_anns.extend(img_id_2_anns[img['id']])

eval_dataset = {
    'info': dataset['info'],
    'licenses': dataset['licenses'],
    'categories': dataset['categories'],
    'images': eval_images,
    'annotations': eval_anns,
}

with open(os.path.join(dataset_root, 'eval.json'), 'w') as f:
    json.dump(eval_dataset, f)

normal_images = []
for id, name in enumerate(os.listdir(os.path.join(dataset_root, 'normal'))):
    img = Image.open(os.path.join(dataset_root, 'normal', name))
    width, height = img.size
    name = os.path.join('..', 'normal', name)
    img = {
        "coco_url": "",
        "data_captured": "",
        "file_name": name,
        "flickr_url": "",
        "id": id + 100000,
        "height": height,
        "width": width,
        "license": 1,
    }
    normal_images.append(img)

train_dataset = {
    'info': dataset['info'],
    'licenses': dataset['licenses'],
    'categories': dataset['categories'],
    'images': train_images,
    'annotations': train_anns,
    'extra_images': normal_images,
}

print(len(normal_images))

with open(os.path.join(dataset_root, 'train.json'), 'w') as f:
    json.dump(train_dataset, f)
