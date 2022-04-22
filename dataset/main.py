from torch.utils import data
from .cifar10 import CIFAR10_Dataset
from .cifar100 import CIFAR100_Dataset
from .mvtec import MVTec_Dataset, CLASS_NAMES
def load_dataset(dataset_name, data_path, normal_class, data_augmentation = True, normalize=True):
    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root = data_path, normal_class = normal_class)

    elif dataset_name == 'cifar100':
        dataset = CIFAR100_Dataset(root = data_path, normal_class = normal_class)
    elif dataset_name == 'mvtec':
        train_dataset = MVTec_Dataset(root = data_path, normal_class = CLASS_NAMES[normal_class], is_train = True)
        test_dataset = MVTec_Dataset(root = data_path, normal_class = CLASS_NAMES[normal_class], is_train = False)
        dataset = (train_dataset, test_dataset)
    return dataset
