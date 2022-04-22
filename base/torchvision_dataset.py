from .base_dataset import BaseADDataset
from torch.utils.data import DataLoader

class TorchvisionDataset(BaseADDataset):
    def __init__(self, root):
        super().__init__(root)
        self.image_size = None  # tuple with the size of an image from the dataset (e.g. (1, 28, 28) for MNIST)
    def loaders(self, batch_size, shuffle_train,  shuffle_test):
        train_loader = DataLoader(dataset = self.train_set, batch_size = batch_size, shuffle = shuffle_train)
        test_loader = DataLoader(dataset = self.test_set, batch_size = batch_size, shuffle = shuffle_test)

        return train_loader, test_loader