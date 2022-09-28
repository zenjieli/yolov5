from email.mime import base
import os
import os.path as osp
import shutil
import numpy as np

def split_dataset(srcdir, destdirs, dir2_ratio):
    """
    Split a dataset in one directory into to directories    
    """
    assert len(destdirs) == 2

    jpg_files = [filename for filename in os.listdir(srcdir) if filename.lower().endswith('.jpg')]
    np.random.shuffle(jpg_files)

    filenames_two_dirs = np.split(np.array(jpg_files), [int(len(jpg_files) * (1 - dir2_ratio))])

    print(f'Number of training images: {len(filenames_two_dirs[0])}; Number of validation images: {len(filenames_two_dirs[1])}')
    for filenames, destdir in zip(filenames_two_dirs, destdirs):
        for filename in filenames:            
            shutil.copy(osp.join(srcdir, filename), destdir)
            txt_filename = osp.splitext(filename)[0] + '.txt'
            shutil.copy(osp.join(srcdir, txt_filename), destdir)

if __name__ == '__main__':
    basedir = osp.expanduser('~/data/person_dataset/')
    split_dataset(basedir + 'milestone', [basedir + 'train', basedir + 'val'], 0.25)