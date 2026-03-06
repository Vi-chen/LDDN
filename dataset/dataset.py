import os
import cv2
import numpy as np
import torch.utils.data as data
from glob import glob


class Dataset(data.Dataset):
    """
    Support two data organization methods:：
    1. List file method:
       data/
         ├── A/
         ├── B/
         ├── label/
         └── list/train.txt

    2. Folder classification method:
       data/
         ├── train/
         │   ├── A/
         │   ├── B/
         │   └── label/
         ├── val/
         └── test/
    """

    def __init__(self, dataset, file_root='data/', transform=None):
        self.dataset = dataset
        self.file_root = file_root
        self.transform = transform

        list_dir = os.path.join(file_root, 'list1')
        self.use_list_mode = os.path.isdir(list_dir)

        if self.use_list_mode:
            print(f"📄 The 'list' folder was detected. Data was loaded using the 『list method』.")
            self._init_from_list()
        else:
            print(f"📂 The "list" folder was not detected. Using the 『folder partitioning method』 to load data.")
            self._init_from_folders()

    # === 1. List file method  ===
    def _init_from_list(self):
        list_path = os.path.join(self.file_root, 'list', f'{self.dataset}.txt')
        if not os.path.isfile(list_path):
            raise FileNotFoundError(f"Can't find {list_path}")
        self.file_list = open(list_path).read().splitlines()

        self.pre_images = [os.path.join(self.file_root, 'A', x) for x in self.file_list]
        self.post_images = [os.path.join(self.file_root, 'B', x) for x in self.file_list]
        self.gts = [os.path.join(self.file_root, 'label', x) for x in self.file_list]
        self.names = self.file_list

    # === 2. Folder classification method ===
    def _init_from_folders(self):
        split_dir = os.path.join(self.file_root, self.dataset)
        dir_t1 = os.path.join(split_dir, 'A')
        dir_t2 = os.path.join(split_dir, 'B')
        dir_lbl = os.path.join(split_dir, 'label')

        for d in [dir_t1, dir_t2, dir_lbl]:
            if not os.path.isdir(d):
                raise FileNotFoundError(f"The directory does not exist: {d}")

        exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        t1_files = [os.path.basename(p) for p in self._list_files(dir_t1, exts)]
        t2_files = [os.path.basename(p) for p in self._list_files(dir_t2, exts)]
        lbl_files = [os.path.basename(p) for p in self._list_files(dir_lbl, exts)]

        common = sorted(list(set(t1_files) & set(t2_files) & set(lbl_files)))
        if not common:
            raise RuntimeError("No matching samples were found. Please check if the file names are consistent.")

        self.pre_images = [os.path.join(dir_t1, n) for n in common]
        self.post_images = [os.path.join(dir_t2, n) for n in common]
        self.gts = [os.path.join(dir_lbl, n) for n in common]
        self.names = common

    @staticmethod
    def _list_files(folder, exts):
        files = []
        for ext in exts:
            files.extend(glob(os.path.join(folder, f'*{ext}')))
        return files

    def __len__(self):
        return len(self.pre_images)

    def __getitem__(self, idx):
        name = self.names[idx]

        pre_image = cv2.imread(self.pre_images[idx], cv2.IMREAD_COLOR)
        post_image = cv2.imread(self.post_images[idx], cv2.IMREAD_COLOR)
        label = cv2.imread(self.gts[idx], cv2.IMREAD_GRAYSCALE)

        if pre_image is None or post_image is None or label is None:
            raise RuntimeError(f"Read failure：{name}")

        img = np.concatenate((pre_image, post_image), axis=2)

        if self.transform:
            img, label = self.transform(img, label)

        return img, label, name
