import cv2
import json
import os.path as osp
import shutil

def get_img_shape(img_path):
    img = cv2.imread(img_path)
    try:
        return img.shape
    except AttributeError:
        print('error!', img_path)
        return (None, None, None)

def convert_labels(img_path, x0, y0, w, h):
    """
    Convert labels (x0, y0, abs_w, abs_h) => (x_center_rel, y_center_rel, w_rel, h_rel)
    See ./sample.json for the COCO annotation format
    """
    size = get_img_shape(img_path)
    dw = 1. / size[1]
    dh = 1. / size[0]
    x = (x0 + x0 + w) / 2.0
    y = (y0 + y0 + h) / 2.0
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def get_image_name_dict(json_data):
    img_filename_dict = {}

    for img_data in json_data['images']:
        if int(img_data['license']) == 4:
            img_filename_dict[img_data['id']] = img_data['file_name']

    return img_filename_dict

def convert_coco_to_yolo(src_img_dir, json_path, dst_dir):
    annotation_key = 'annotations'
    img_id = 'image_id'
    cat_id = 'category_id'
    bbox_id = 'bbox'

    # Enter directory to read JSON file
    data = json.load(open(json_path))

    img_filename_dict = get_image_name_dict(data)

    check_set = set()

    # Retrieve data
    for anno in data[annotation_key]:
        # Get required data
        image_id = anno[img_id]
        category_id = int(anno[cat_id])
        if category_id != 1 or not (image_id in img_filename_dict):  # Not person
            continue

        bbox = anno[bbox_id]
        img_filename = img_filename_dict[image_id]
        image_path = osp.join(src_img_dir, img_filename)
        label_path = osp.join(dst_dir, osp.splitext(img_filename)[0] + '.txt')

        # Convert the data
        yolo_bbox = convert_labels(image_path, bbox[0], bbox[1], bbox[2], bbox[3])

        # Prepare for export
        content = f"{0} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"

        # Export
        if image_id in check_set:
            # Append to existing file as there can be more than one label in each image
            file = open(label_path, 'a')
            file.write('\n')
            file.write(content)
            file.close()

        elif image_id not in check_set:
            check_set.add(image_id)
            # Copy image file
            shutil.copy(image_path, dst_dir)

            # Write label
            file = open(label_path, 'w')
            file.write(content)
            file.close()


# To run in as a class
if __name__ == "__main__":
    convert_coco_to_yolo(osp.expanduser('/home/zli/data/coco-2017/validation/data'),
                      osp.expanduser('/home/zli/data/coco-2017/validation/labels.json'),
                      osp.expanduser('~/data/person_mix/images/val'))
