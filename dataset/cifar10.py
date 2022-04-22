from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import CIFAR10
from base.torchvision_dataset import TorchvisionDataset
import torchvision.transforms as transforms
import torch
import numpy as np
class CIFAR10_Dataset(TorchvisionDataset):

    def __init__(self, root, normal_class = 5):
        super().__init__(root)

        self.n_classes = 2
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)
        self.outlier_classes = tuple(self.outlier_classes)

        data_transform = transforms.Compose([
                transforms.Resize((256, 256), Image.ANTIALIAS),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set = CIFAR10(root=self.root, train=True, transform=data_transform, target_transform=target_transform,
                              download=True)
        self.test_set = CIFAR10(root=self.root, train=False, transform=data_transform,
                                  target_transform=target_transform, download=True)
        idx = np.argwhere(np.isin(np.array(train_set.targets), self.normal_classes))
        idx = idx.flatten().tolist()
        self.train_set = Subset(train_set, idx)


class MyCIFAR10(CIFAR10):
    """Torchvision CIFAR10 class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super(MyCIFAR10, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """Override the original method of the CIFAR10 class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index
class CIFAR10_Dataset2(TorchvisionDataset):

    def __init__(self, root, normal_class = 5):
        super().__init__(root)

        self.n_classes = 2
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)
        self.outlier_classes = tuple(self.outlier_classes)

        data_transform = transforms.Compose([
                transforms.Resize((256, 256), Image.ANTIALIAS),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set = MyCIFAR10(root=self.root, train=True, transform=data_transform, target_transform=target_transform,
                              download=True)
        self.test_set = MyCIFAR10(root=self.root, train=False, transform=data_transform,
                                  target_transform=target_transform, download=True)
        idx = np.argwhere(np.isin(np.array(train_set.targets), self.normal_classes))
        idx = idx.flatten().tolist()
        self.train_set = Subset(train_set, idx)
        # return img, target, index


