import os, sys, tempfile
import simplejson as json

this_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(this_dir, '..', '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.build import build_dataset
from maskrcnn_benchmark.utils.imports import import_file

def main():
    cfg.freeze()

    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset = build_dataset(['tianchi_xray_eval_no_normal_bbox_in_coco'], None, DatasetCatalog, False, False)[0]
    id_to_img_map = dataset.id_to_img_map
    print(json.dumps(id_to_img_map))


if __name__ == '__main__':
    main()
