import os
import os.path as osp
import shutil


def copy_imgs_incomplete_persons(src_dir, dst_dir, ref_dir):
    for dirpath, _, files in os.walk(src_dir):
        for filename in files:
            if osp.splitext(filename)[-1].lower() != '.jpg':
                continue

            # No person detected by ref detector
            if not osp.exists(osp.join(ref_dir, osp.splitext(filename)[0] + '.txt')):
                shutil.copy(osp.join(dirpath, filename), dst_dir)

        # Don't search subdirectories
        break


def copy_imgs_complete_persons(src_dir, dst_dir, imgs_to_remove_dir):
    for dirpath, _, files in os.walk(src_dir):
        for filename in files:
            if osp.splitext(filename)[-1].lower() != '.jpg':
                continue

            # Check if this image should be copied
            if not osp.exists(osp.join(imgs_to_remove_dir, filename)):
                shutil.copy(osp.join(dirpath, filename), dst_dir)
                shutil.copy(osp.join(dirpath, osp.splitext(filename)[0] + '.txt'), dst_dir)

        # Don't search subdirectories
        break


def copy_manual_labels(work_dir, label_dir1, label_dir2):
    for dirpath, _, files in os.walk(work_dir):
        for filename in files:
            if osp.splitext(filename)[-1].lower() != '.txt':
                continue

            # Check if manual label exists
            if osp.exists(osp.join(label_dir1, filename)):
                shutil.copy(osp.join(label_dir1, filename), dirpath)
            elif osp.exists(osp.join(label_dir2, filename)):
                shutil.copy(osp.join(label_dir2, filename), dirpath)

        # Don't search subdirectories
        break


if __name__ == '__main__':
    copy_manual_labels(osp.expanduser('~/data/person_mix/images/val'),
                       osp.expanduser('~/data/person_cleansed/images/train'),
                       osp.expanduser('~/data/person_cleansed/images/val'))
    # copy_imgs_complete_persons(osp.expanduser('~/data/person_mix_full/images/val'),
    #                         osp.expanduser('~/data/person_mix/images/val'),
    #                         osp.expanduser('~/data/person_mix/images/valToRemove'))
    # copy_imgs_incomplete_persons(osp.expanduser('~/data/person_mix_full/images/val'),
    #                              osp.expanduser('~/data/person_mix/images/val'),
    #                              osp.expanduser('~/workspace/github/yolov5/runs/detect/exp4/labels'))
