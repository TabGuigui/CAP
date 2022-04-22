import os
import tarfile
from PIL import Image
from tqdm import tqdm
import urllib.request

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', # MVTEC data classes
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper'] 

class MVTec_Dataset(Dataset):
    def __init__(self, root = './data', normal_class = 'bottle', is_train = True):
        assert normal_class in CLASS_NAMES, 'class_name: {}, should be in {}'.format(normal_class, CLASS_NAMES)
        self.root_path = root
        self.normal_class = normal_class
        self.is_train = is_train
        self.mvtec_folder_path = os.path.join(self.root_path, 'mvtec_anomaly_detection')

        self.download()

        self.x, self.y, self.mask = self.load_dataset_folder()
        self.transform_x = T.Compose([T.Resize(256, Image.ANTIALIAS),
                                      T.CenterCrop(224),
                                      T.ToTensor(),
                                      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transform_mask = T.Compose([T.Resize(256, Image.NEAREST),
                                         T.CenterCrop(224),
                                         T.ToTensor()])
    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if y == 0:
            mask = torch.zeros([1, 224, 224])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y , mask
    def __len__(self):
        return len(self.x)
    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []
        img_dir = os.path.join(self.mvtec_folder_path, self.normal_class, phase)
        gt_dir = os.path.join(self.mvtec_folder_path, self.normal_class, 'ground_truth')

        img_types = sorted(os.listdir(img_dir)) # train good  # test brokenlarge brokensmall contamination

        for img_type in img_types:
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
                
            img_fpath_list = sorted([os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.png')])
            x.extend(img_fpath_list) # path

            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)
        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)
    def download(self):

        if not os.path.exists(self.mvtec_folder_path):
            tar_file_path = self.mvtec_folder_path + '.tar.xz' # zip
            if not os.path.exists(tar_file_path):
                download_url(URL, tar_file_path)
            print('unzip download dataset')
            tar =tarfile.open(tar_file_path, 'r:xz')
            tar.extractall(self.mvtec_folder_path)
            tar.close()

        return 

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

