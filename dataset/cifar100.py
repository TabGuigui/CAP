from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import CIFAR100
from base.torchvision_dataset import TorchvisionDataset
import torchvision.transforms as transforms
import torch
import numpy as np

class CIFAR100_Dataset(TorchvisionDataset):

    def __init__(self, root, normal_class = 5):
        super().__init__(root)

        self.n_classes = 2
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 20))
        self.outlier_classes.remove(normal_class)
        self.outlier_classes = tuple(self.outlier_classes)

        data_transform = transforms.Compose([
                transforms.Resize((256, 256), Image.ANTIALIAS),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set = CIFAR100Coarse(root=self.root, train=True, transform=data_transform, target_transform=target_transform,
                              download=True)
        self.test_set = CIFAR100Coarse(root=self.root, train=False, transform=data_transform,
                                  target_transform=target_transform, download=True)
        idx = np.argwhere(np.isin(np.array(train_set.targets), self.normal_classes))
        idx = idx.flatten().tolist()
        self.train_set = Subset(train_set, idx)

class CIFAR100Coarse(CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100Coarse, self).__init__(root, train, transform, target_transform, download)

        # update labels
        coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                   3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                   6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
                                   0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                   5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                   16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                                   10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
                                   2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                  16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                  18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
        self.targets = coarse_labels[self.targets]
