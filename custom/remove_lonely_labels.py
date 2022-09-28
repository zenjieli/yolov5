import os
import os.path as osp

def remove_lonely_labels(image_label_path):
    '''
    image_label_path: Path for the directory containing both images and labels
    '''
    for root, _, filenames in os.walk(image_label_path):
        for f in filenames:
            name_without_ext, ext = os.path.splitext(f)
            if ext.lower() == '.txt' and not os.path.exists(os.path.join(root, name_without_ext + '.jpg')):
                os.remove(os.path.join(root, f))

        break

if __name__ == '__main__':
    remove_lonely_labels(osp.expanduser('~/data/person_dataset/images/val'))
