import simplejson as json
import os
from PIL import Image


def main():
    objs = {}
    with open('./datasets/tianchi_xray/train_no_poly.json') as f:
        objs = json.load(f)

    new_objs = {}
    for key in objs.keys():
        if key in ['info', 'licenses', 'categories']:
            new_objs[key] = objs[key]


    images = []
    for id, name in enumerate(os.listdir('./datasets/tianchi_xray/test_a')):
        path = os.path.join('./datasets/tianchi_xray/test_a', name)
        img = Image.open(path)
        width, height = img.size
        entry = {
            "coco_url": "",
            "data_captured": "",
            "file_name": name,
            "flickr_url": "",
            "id": id,
            "height": height,
            "width": width,
            "license": 1,
        }
        images.append(entry)
    new_objs['images'] = images

    with open('./datasets/tianchi_xray/train_no_poly_test_a.json', 'w') as f:
        json.dump(new_objs, f)


if __name__ == '__main__':
    main()
