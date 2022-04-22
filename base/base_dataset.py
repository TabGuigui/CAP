from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

class BaseADDataset(ABC):

    def __init__(self, root):
        super().__init__()

        self.root = root
        self.n_classes = 2 
        self.normal_classes = None
        self.outlier_classes = None

        self.train_set = None
        self.test_set = None

    @abstractmethod
    def loaders(self, batch_size, shuffle_train = True, shuffle_test = False):
        pass

    def __repr__(self):
        return self.__class__.__name__