import argparse
import os, sys, tempfile
import torch

this_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(this_dir, '..', '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.data.datasets.evaluation.coco.coco_eval import (
    check_expected_results,
    prepare_for_coco_detection,
    evaluate_box_proposals,
    evaluate_predictions_on_coco,
    COCOResults,
)
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.layers import nms as _box_nms
from maskrcnn_benchmark.layers import soft_nms as _box_soft_nms


parser = argparse.ArgumentParser(description="PyTorch Object Detection Eval")
parser.add_argument(
    "--config-file",
    default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
    metavar="FILE",
    help="path to config file",
)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--no-eval", default=False, action="store_true")
parser.add_argument(
    "opts",
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)

args = parser.parse_args()

cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()

data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=False)

dataset_names = cfg.DATASETS.TEST
output_folders = [None] * len(cfg.DATASETS.TEST)
if cfg.OUTPUT_DIR:
    for idx, dataset_name in enumerate(dataset_names):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        mkdir(output_folder)
        output_folders[idx] = output_folder

extra_args = dict(
    box_only=False,
    iou_types=("bbox",),
    expected_results=cfg.TEST.EXPECTED_RESULTS,
    expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
)

# preds = torch.load(os.path.join(cfg.OUTPUT_DIR, "inference", "tianchi_xray_eval_no_normal_bbox_in_coco", "predictions.pth"))
base_path = 'datasets/tianchi_xray/pred/ms'
pred_1 = torch.load(os.path.join(base_path, 'predictions.pth'))
pred_2 = torch.load(os.path.join(base_path, 'predictions_640.pth'))
pred_3 = torch.load(os.path.join(base_path, 'predictions_960.pth'))
pred_4 = torch.load(os.path.join(base_path, 'predictions_1280.pth'))


def do_coco_evaluation(
    dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    print("Preparing results for COCO format")
    coco_results = {}
    if "bbox" in iou_types:
        print("Preparing bbox results")
        coco_results["bbox"] = prepare_for_coco_detection(predictions, dataset)

    results = COCOResults(*iou_types)
    print("Evaluating predictions")
    for iou_type in iou_types:
        with tempfile.NamedTemporaryFile() as f:
            file_path = f.name
            if output_folder:
                file_path = os.path.join(output_folder, iou_type + ".json")
            res = evaluate_predictions_on_coco(
                dataset.coco, coco_results[iou_type], file_path, iou_type
            )
            results.update(res)
    print(results)
    check_expected_results(results, expected_results, expected_results_sigma_tol)
    return results, coco_results


def preds_filter(preds, limit):
    cnt1 = 0
    cnt2 = 0
    for pred in preds:
        extra_fields = pred.extra_fields
        selected_ids = []
        for idx, score in enumerate(extra_fields['scores']):
            if score > limit and extra_fields['labels'][idx] in range(1, 6):
                selected_ids.append(idx)

        extra_fields['scores'] = extra_fields['scores'][selected_ids]
        extra_fields['labels'] = extra_fields['labels'][selected_ids]
        cnt1 = cnt1 + pred.bbox.size(0)
        pred.bbox = pred.bbox[selected_ids]
        cnt2 = cnt2 + pred.bbox.size(0)

    print('bbox: ', cnt1, 'to', cnt2)
    return preds


def cat_boxlist(boxlists):
    new_boxlists = []
    for arr in zip(*boxlists):
        size = arr[0].size
        mode = arr[0].mode
        bboxes = []
        scores = []
        labels = []
        for boxlist in arr:
            bboxes.append(boxlist.bbox)
            scores.append(boxlist.extra_fields['scores'])
            labels.append(boxlist.extra_fields['labels'])
        bboxes = torch.cat(bboxes)
        scores = torch.cat(scores)
        labels = torch.cat(labels)
        boxlist = BoxList(bboxes, size, mode)
        boxlist.extra_fields['scores'] = scores
        boxlist.extra_fields['labels'] = labels
        new_boxlists.append(boxlist)
    return new_boxlists


def perform_nms(preds, nms_thresh):
    new_preds = []
    for idx, boxlist in enumerate(preds):
        new_bbox = []
        new_scores = []
        new_labels = []
        labels = boxlist.extra_fields['labels']
        scores = boxlist.extra_fields['scores']
        for i in range(1, 6):
            bbox = boxlist.bbox
            inds = labels == i
            bbox_i = bbox[inds]
            labels_i = labels[inds]
            scores_i = scores[inds]
            # keep = _box_nms(bbox_i, scores, nms_thresh)
            keep = _box_soft_nms(bbox_i, scores_i, nms_thresh, 0, 0.5, 0.001)
            new_bbox.append(bbox_i[keep])
            new_scores.append(scores_i[keep])
            new_labels.append(labels_i[keep])
        new_bbox = torch.cat(new_bbox)
        new_scores = torch.cat(new_scores)
        new_labels = torch.cat(new_labels)
        new_boxlist = BoxList(new_bbox, boxlist.size, boxlist.mode)
        new_boxlist.extra_fields['labels'] = new_labels
        new_boxlist.extra_fields['scores'] = new_scores
        if new_bbox.size(0) > 100:
            image_thresh, _ = torch.kthvalue(
                new_scores, new_bbox.size(0) - 100 + 1
            )
            keep = new_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            new_boxlist = new_boxlist[keep]
        new_preds.append(new_boxlist)
    return new_preds


def main():
    preds_filter(pred_1, 0.05)
    preds_filter(pred_2, 0.05)
    # preds_filter(pred_3, 0.05)
    # preds_filter(pred_4, 0.05)

    preds = cat_boxlist((pred_2, pred_1))
    preds = perform_nms(preds, 0.5)

    do_coco_evaluation(
        dataset=data_loaders_val[0].dataset,
        predictions=preds,
        output_folder=output_folders[0],
        **extra_args
    )


if __name__ == '__main__':
    main()
