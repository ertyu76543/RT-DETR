from pycocotools.coco import COCO

# 그래프
# import matplotlib.pyplot as plt
# import numpy as np


annotation_file = '/mnt/share/lvis/lvis_v1_train.json'

coco = COCO(annotation_file)

ids = list(sorted(coco.imgs.keys())) 

cat_ids = coco.getCatIds()

cat_data = coco.loadCats(cat_ids)

# for idx, _cat_data in enumerate(cat_data):
#     image_count[idx] = _cat_data["image_count"]
#     instance_counts[idx] = _cat_data["instance_count"]

