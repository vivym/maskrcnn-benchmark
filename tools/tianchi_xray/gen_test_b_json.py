import os
import simplejson as json
from PIL import Image


def main():
    with open('./datasets/tianchi_xray/test_no_poly_a.json') as f:
        test_a = json.load(f)

    images = []
    for idx, filename in enumerate(os.listdir('./datasets/tianchi_xray/test_b/')):
        img = Image.open(os.path.join('datasets', 'tianchi_xray', 'test_b', filename))
        width, height = img.size
        entry = {
            "coco_url": "",
            "data_captured": "",
            "file_name": filename,
            "flickr_url": "",
            "id": idx,
            "height": height,
            "width": width,
            "license": 1,
        }
        images.append(entry)
    test_a['images'] = images

    with open('./datasets/tianchi_xray/test_no_poly_b.json', 'w') as f:
        json.dump(test_a, f)


if __name__ == '__main__':
    main()
