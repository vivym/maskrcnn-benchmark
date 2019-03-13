import simplejson as json
import random


def main():
    objs = {}
    with open('./datasets/tianchi_xray/train_no_poly_adapted.json') as f:
        objs = json.load(f)

    images = objs['images']
    annotations = objs['annotations']
    random.shuffle(images)

    eval_images = images[:100]
    train_images = images[100:]

    anns_set = {}
    for ann in annotations:
        id = ann['image_id']
        if id not in anns_set:
            anns_set[id] = []
        anns_set[id].append(ann)

    new_objs = {}
    for k in objs.keys():
        if k not in ['images', 'annotations']:
            new_objs[k] = objs[k]

    with open('./datasets/tianchi_xray/train_no_poly_adapted_ex_eval.json', 'w') as f:
        new_objs['images'] = train_images
        anns = []
        for img in train_images:
            img_id = img['id']
            if img_id in anns_set:
                anns.extend(anns_set[img_id])
        new_objs['annotations'] = anns
        json.dump(new_objs, f)

    with open('./datasets/tianchi_xray/eval_no_poly_adapted.json', 'w') as f:
        new_objs['images'] = eval_images
        anns = []
        for img in eval_images:
            img_id = img['id']
            if img_id in anns_set:
                anns.extend(anns_set[img_id])
        new_objs['annotations'] = anns
        json.dump(new_objs, f)


def check():
    objs = {}
    with open('./datasets/tianchi_xray/train_no_poly_adapted.json') as f:
        objs = json.load(f)

    train = {}
    with open('./datasets/tianchi_xray/train_no_poly_adapted_ex_eval.json') as f:
        train = json.load(f)

    eval = {}
    with open('./datasets/tianchi_xray/eval_no_poly_adapted.json') as f:
        eval = json.load(f)

    cnt = 0
    for idx, entry in enumerate(train['images']):
        id = entry['id']

        for i, img in enumerate(objs['images']):
            if img['id'] == id:
                if train['annotations'][idx]['category_id'] != objs['annotations'][i]['category_id']:
                    print(entry)
                else:
                    cnt = cnt + 1
    print(cnt)
    cnt = 0
    for idx, entry in enumerate(eval['images']):
        id = entry['id']

        for i, img in enumerate(objs['images']):
            if img['id'] == id:
                eval_ann = eval['annotations'][idx]
                objs_ann = objs['annotations'][i]
                if eval_ann['category_id'] != objs_ann['category_id']:
                    print(entry)
                elif eval_ann['image_id'] != id:
                    print(entry)
                else:
                    cnt = cnt + 1
    print(cnt)


def main_100():
    objs = {}
    with open('./datasets/tianchi_xray/train_no_poly_adapted.json') as f:
        objs = json.load(f)

    images = objs['images']
    annotations = objs['annotations']
    entries = list(zip(images, annotations))

    eval = entries[:100]

    eval_images = []
    eval_ann = []
    for (img, ann) in eval:
        eval_images.append(img)
        eval_ann.append(ann)

    with open('./datasets/tianchi_xray/train_no_poly_100.json', 'w') as f:
        objs['images'] = eval_images
        objs['annotations'] = eval_ann
        json.dump(objs, f)


if __name__ == '__main__':
    main()
