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

preds = torch.load(os.path.join(cfg.OUTPUT_DIR, "inference", "tianchi_xray_eval_no_normal_bbox_in_coco", "predictions.pth"))


def do_coco_evaluation(
    dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    if box_only:
        print("Evaluating bbox proposals")
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        res = COCOResults("box_proposal")
        for limit in [100, 1000]:
            for area, suffix in areas.items():
                stats = evaluate_box_proposals(
                    predictions, dataset, area=area, limit=limit
                )
                key = "AR{}@{:d}".format(suffix, limit)
                res.results["box_proposal"][key] = stats["ar"].item()
        print(res)
        check_expected_results(res, expected_results, expected_results_sigma_tol)
        if output_folder:
            torch.save(res, os.path.join(output_folder, "box_proposals.pth"))
        return
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
        deleted_ids = []
        for idx, score in enumerate(extra_fields['scores']):
            if score < limit:
                deleted_ids.append(idx)
            elif extra_fields['labels'][idx] not in range(1, 6):
                deleted_ids.append(idx)

        extra_fields['scores'] = list(filter(lambda x: x[0] not in deleted_ids, enumerate(extra_fields['scores'])))
        extra_fields['scores'] = list(map(lambda x: x[1], extra_fields['scores']))
        extra_fields['scores'] = torch.FloatTensor(extra_fields['scores'])
        extra_fields['labels'] = list(filter(lambda x: x[0] not in deleted_ids, enumerate(extra_fields['labels'])))
        extra_fields['labels'] = list(map(lambda x: x[1], extra_fields['labels']))
        extra_fields['labels'] = torch.tensor(extra_fields['labels'])
        cnt1 = cnt1 + pred.bbox.size(0)
        pred.bbox = list(filter(lambda x: x[0] not in deleted_ids, enumerate(pred.bbox.tolist())))
        pred.bbox = list(map(lambda x: x[1], pred.bbox))
        pred.bbox = torch.FloatTensor(pred.bbox)
        cnt2 = cnt2 + pred.bbox.size(0)

    print('bbox: ', cnt1, 'to', cnt2)
    return preds


def main():
    preds_filter(preds, 0.05)

    do_coco_evaluation(
        dataset=data_loaders_val[0].dataset,
        predictions=preds,
        output_folder=output_folders[0],
        **extra_args
    )


if __name__ == '__main__':
    main()
