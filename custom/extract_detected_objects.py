import os.path as osp
import os
import PIL.Image as Image


def crop_and_save_detected_objects(img_filepath, label_filepath, out_dir, dilation_ratio=1):
    basename = osp.splitext(osp.basename(img_filepath))[0]
    img = Image.open(img_filepath, 'r')

    label_file = open(label_filepath, 'r')
    i = 0
    for line in label_file:
        words = line.split()
        patch_w = float(words[3]) * img.width
        patch_h = float(words[4]) * img.height
        patch_w *= dilation_ratio
        patch_h *= dilation_ratio
        patch_l = float(words[1]) * img.width - patch_w / 2
        patch_t = float(words[2]) * img.height - patch_h / 2

        patch_r = max(0, min(img.width, patch_l + patch_w))
        patch_d = max(0, min(img.height, patch_t + patch_h))
        patch_l = max(0, min(img.width - 1, patch_l))
        patch_t = max(0, min(img.height - 1, patch_t))
        img.crop((patch_l, patch_t, patch_r, patch_d)).save(osp.join(out_dir, basename + f'_{i}.jpg'))
        i += 1


def extract_detected_objects(img_dir, label_dir, out_dir, dilation_ratio):
    img_files = [f for f in os.listdir(img_dir) if f.lower()[-4:] in ['.jpg', '.bmp']]

    for img_filename in img_files:
        label_filepath = osp.join(label_dir, osp.splitext(img_filename)[0] + '.txt')
        if osp.exists(label_filepath):  # Something has been detected
            crop_and_save_detected_objects(osp.join(img_dir, img_filename), label_filepath, out_dir, dilation_ratio)

    return 0


if __name__ == '__main__':
    label_dir = osp.expanduser('~/workspace/github/yolov5/runs/detect/exp/labels')
    img_dir = osp.expanduser('~/Downloads/extracted_images')
    out_dir = osp.expanduser('~/Downloads/extracted_images_out')

    extract_detected_objects(img_dir, label_dir, out_dir, dilation_ratio=2)
