import json
from logging import exception
import os
import os.path as osp

def check_licenses(json_path, img_dir):
    """
    Check if all images have CC BY licenses
    """

    img_count = 0

    data = json.load(open(json_path))
    img_filename_dict = {}
    for img_data in data['images']:
        img_filename_dict[img_data['file_name']] = int(img_data['license'])

    for _, _, files in os.walk(img_dir):
        for filename in files:
            if osp.splitext(filename)[-1].lower() != '.jpg':
                continue

            if img_filename_dict[filename] != 4:
                raise Exception(f'{filename} has wrong license type: {img_filename_dict[filename]}')
            else:
                img_count += 1

    print(f'All {img_count} images have the correct licenses')


if __name__ == '__main__':
    check_licenses('/home/zli/data/coco-2017/validation/labels.json', '/home/zli/data/person_mix/images/val')
