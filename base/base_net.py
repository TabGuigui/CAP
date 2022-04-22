import logging
import torch.nn as nn
import numpy as np

class BaseNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rep_dim = None

    def forward(self, *input):

        raise NotImplementedError

    def summary(self):
        net_parameters = filter(lambda p: p.requires_grad, self.parameters)
        params = sum([np.prod(p.size()) for p in net_parameters])
        self.logger.info('trainable parameters {}'.format(params))
        self.logger.info(self)