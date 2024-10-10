from pycocotools.coco import COCO

# 그래프
# import matplotlib.pyplot as plt
# import numpy as np


annotation_file = '/mnt/share/lvis/lvis_v1_train.json'

coco = COCO(annotation_file)

ids = list(sorted(coco.imgs.keys())) 


def replicate_ri_values_to_list(input_filename, output_filename, target_ids):
    # Create a set of target IDs for faster lookup
    target_ids = target_ids
    output_list = []

    # Open and read the input file line by line
    with open(input_filename, 'r') as file:
        for line in file:
            # Split the line to extract image ID and r_i value
            if 'r_i' in line:
                parts = line.split(', r_i: ')
                image_id = parts[0].strip()  # Extract and clean image ID
                ri_value = int(parts[1].strip())  # Extract and convert r_i to an integer

                # Check if the image_id is in the target IDs list
                if int(image_id) in target_ids:
                    # Replicate the line 'ri_value' times and add it to the output
                    output_list.extend([int(image_id)] * ri_value)

     # Write the output list to a file
    with open(output_filename, 'w') as output_file:
        for image_id in output_list:
            output_file.write(f"{image_id}\n")

# Example usage with input filename and target IDs
input_filename = '/EFL_test/RT-DETR-main/rtdetr_pytorch/rounded_ri_values.txt'
output_filename = '/EFL_test/RT-DETR-main/rtdetr_pytorch/replicated_ri_values.txt'
target_ids = ids

result = replicate_ri_values_to_list(input_filename, output_filename, target_ids)



