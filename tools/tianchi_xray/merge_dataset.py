import simplejson as json


def main():
    with open('./datasets/tianchi_xray/train_no_poly_adapted_ex_eval_with_normal.json') as f:
        train_objs = json.load(f)
    with open('./datasets/tianchi_xray/eval_no_poly_adapted_no_normal_bbox_with_area.json') as f:
        test_objs = json.load(f)

    print(train_objs.keys())
    new_images = []
    for entry in test_objs['images']:
        id = entry['id']
        file_name = entry['file_name']
        if id < 100000:
            new_images.append(entry)

    new_anns = []
    for entry in test_objs['annotations']:
        image_id = entry['image_id']
        if image_id < 100000:
            new_anns.append(entry)

    cnt = 0
    for entry in train_objs['images']:
        id = entry['id']
        if id < 100000:
            cnt = cnt + 1
    print(cnt)


    print(len(new_images), len(new_anns), len(train_objs['images']), len(train_objs['annotations']))

    train_objs['images'].extend(new_images)
    train_objs['annotations'].extend(new_anns)

    print(len(train_objs['images']), len(train_objs['annotations']))

    with open('./datasets/tianchi_xray/train_no_poly_adapted_full_with_normal.json', 'w') as f:
        json.dump(train_objs, f)


if __name__ == '__main__':
    main()
