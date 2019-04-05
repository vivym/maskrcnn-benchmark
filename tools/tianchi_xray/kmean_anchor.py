import os, sys
import numpy as np
import torch
import torchvision

this_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(this_dir, '..', '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from maskrcnn_benchmark.structures.bounding_box import BoxList

def fill_truth_detection(bs, flip, dx, dy, sx, sy):
    new_bs = []
    # print(bs)
    for i in range(bs.shape[0]):
        x1 = bs[i][0]
        y1 = bs[i][1]
        x2 = bs[i][0] + bs[i][2]
        y2 = bs[i][1] + bs[i][3]

        x1 = min(0.99, max(0, x1 * sx - dx))
        y1 = min(0.99, max(0, y1 * sy - dy))

        x2 = max(0, min(0.999, x2 * sx - dx))
        y2 = max(0, min(0.999, y2 * sy - dy))

        bs[i][0] = x1
        bs[i][1] = y1
        bs[i][2] = x2 - x1
        bs[i][3] = y2 - y1
        bs[i][4] = bs[i][4]

        if flip:
            bs[i][0] = 1 - bs[i][0] - bs[i][2]

        if bs[i][2] > 0 and bs[i][3] > 0:
            new_bs.append([bs[i]])

    new_bs = np.array(new_bs)
    new_bs = np.reshape(new_bs, (-1, 5))

    return new_bs


def norm_bb(b, size):
    x = b[:, 0:1]
    y = b[:, 1:2]

    dw = 1. / size[0]
    dh = 1. / size[1]

    x = (x * dw)
    y = (y * dh)
    w = ((b[:, 2:3] - b[:, 0:1]) * dw)
    h = ((b[:, 3:4] - b[:, 1:2]) * dh)

    return np.concatenate((x, y, w, h, b[:, 4:5]), axis=1)


def load_data_detection(img, bs, shape, aug=True, jitter=0.2, hue=0.1, saturation=1.5, exposure=1.5):
    bs = norm_bb(bs, (img.width, img.height))
    flip, dx, dy, sx, sy = False, 0, 0, 1, 1
    label = fill_truth_detection(bs, flip, dx, dy, 1. / sx, 1. / sy)

    return img, label


min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False

class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, train, transforms=None,box_encoder=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms
        self.train = train
        self.box_encoder = box_encoder


    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        src_img_size = img.size
        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        ##注意类别！！！！！！！！！！！！！
        ##注意类别！！！！！！！！！！！！！
        ##注意类别！！！！！！！！！！！！！
        ##注意类别！！！！！！！！！！！！！
        classes = [self.json_category_id_to_contiguous_id[c]-1 for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        box_coord = target.bbox
        box_cls = target.get_field('labels').view(-1, 1).float()

        bbox_info = torch.cat((box_coord,box_cls),1)


        img, bbox = load_data_detection(img, bbox_info.numpy(), self.transforms.transforms[0].size, self.train)


        if self.box_encoder is not None:
            gt = self.box_encoder(bbox)
        else:
            gt = np.zeros((50, 5), dtype=np.float32)
            gt[:len(bbox), :] = bbox
            gt = torch.from_numpy(gt).float()

        if self.transforms is not None:
            img = self.transforms(img)

        if self.train:
            return img, gt
        else:
            return img,gt,anno[0]['image_id'],src_img_size

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data


def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters


def norm_bb(b, size):
    x = b[:, 0:1]
    y = b[:, 1:2]

    dw = 1. / size[0]
    dh = 1. / size[1]

    x = (x * dw)  # .clip(0.01, 0.99)
    y = (y * dh)  # .clip(0.01, 0.99)
    w = ((b[:, 2:3] - b[:, 0:1]) * dw)  # .clip(0.01, 0.99)
    h = ((b[:, 3:4] - b[:, 1:2]) * dh)  # .clip(0.01, 0.99)
    # b[:, 4:5] = 1

    return np.concatenate((x, y, w, h, b[:, 4:5]), axis=1)


if __name__ == '__main__':
    from torch.utils import data
    import torch

    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize([416, 416]),
        transforms.ToTensor(),
    ])
    datasets_path = 'datasets/tianchi_xray/'

    dataset = COCODataset('{}train_no_poly.json'.format(datasets_path), '{}restricted/'.format(datasets_path), True, False,
                          transform)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=0)
    data = torch.zeros((1, 5))
    for img, label, _, _ in data_loader:
        label_mask = label.sum(-1) > 0
        data = torch.cat((data, label[label_mask]), 0)

    data = data[1:, :]
    data = data[:, 2:4]

    data = data.numpy()

    out = kmeans(data, k=3)
    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
    print("Boxes:\n {}".format(out))

    ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Ratios:\n {}".format(sorted(ratios)))

    out = np.array(out)

    dtype = [('w', float), ('h', float), ('s', float)]
    values = []

    for w,h in out:
        values.append ((w, h ,w*h))

    a = np.array(values, dtype=dtype)  # create a structured array
    a = np.sort(a, order='s')

    print(a)
