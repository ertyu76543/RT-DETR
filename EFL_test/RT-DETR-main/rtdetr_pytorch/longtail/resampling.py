import numpy as np
from collections import defaultdict
from pycocotools.coco import COCO

class IRFS:
    def __init__(self, coco, t):
        self.lvis = coco
        self.imgToCat = defaultdict(list)
        self.cat_ids = self.lvis.getCatIds()
        self.instance_counts = defaultdict(int)
        self.image_counts = defaultdict(int)
        self.total_instance = 0
        self.total_images = len(list(self.lvis.imgs.keys()))
        self.t = t
        
        # image_counts, instance_counts, total_instance, total_images의 값들을 구한다.
        cat_data = self.lvis.loadCats(self.cat_ids)
        for idx, _cat_data in enumerate(cat_data):
            self.image_counts[idx + 1] = _cat_data.get("image_count", 0)
            self.instance_counts[idx + 1] = _cat_data.get("instance_count", 0)
            self.total_instance += int(_cat_data.get("instance_count", 0))
 
        # 이미지마다 어떤 class가 있는 지 파악하는 imgToCat
        for ann in self.lvis.dataset['annotations']:
            self.imgToCat[ann['image_id']].append(ann['category_id'])

    def calculate_f_ic_and_f_bc(self):
        f_ic = {class_id: self.image_counts[class_id] / self.total_images for class_id in self.instance_counts}
        f_bc = {class_id: self.instance_counts[class_id] / self.total_instance for class_id in self.instance_counts}
        return f_ic, f_bc

    def calculate_r_c(self, f_ic, f_bc):
        r_c_values = {}
        for class_id in f_ic:
            value = np.sqrt(self.t / np.sqrt(f_ic[class_id] * f_bc[class_id]))
            r_c = max(1, value)
            r_c_values[class_id] = r_c
        return r_c_values

    def calculate_r_i(self, r_c_values):
        id_list = []
        for image_path, class_ids in self.imgToCat.items():
            max_r_c = max(r_c_values[class_id] for class_id in class_ids)
            id_list.extend([image_path] * round(max_r_c))
        return id_list