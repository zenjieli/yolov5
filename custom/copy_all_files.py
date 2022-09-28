import shutil
import os.path as osp

def main(txt_filepath_with_filenames, src_dir, dest_dir):
    """Copy a list files fron source to destination. The file names are stored in a given text file
    """

    f = open(txt_filepath_with_filenames)
    for filename in f:
        shutil.copy(osp.join(src_dir, filename.strip()), dest_dir)

if __name__ == '__main__':
    txt_filepath_with_filenames = '/home/zli/data/person_dataset_new/files_to_copy.txt'
    src_dir = '/home/zli/data/2022-06-03_1/images1/'
    dest_dir = '/home/zli/data/person_dataset_new/images/'
    main(txt_filepath_with_filenames, src_dir, dest_dir)
