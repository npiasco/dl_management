import setlog
import torch.nn.functional as func
import torch.nn as nn
import torch
import torch.autograd as auto


logger = setlog.get_logger(__name__)


class RMAC(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.R = kwargs.pop('R', 1)  # R=1, Global Max pooling
        self.norm = kwargs.pop('norm', True)
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

    def forward(self, feature):
        x = func.adaptive_max_pool2d(feature, (self.R, self.R))
        x = x.view(x.size(0), -1)
        if self.norm:
            x = func.normalize(x)

        return x


class RAAC(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.R = kwargs.pop('R', 1)  # R=1, Global Max pooling
        self.norm = kwargs.pop('norm', True)
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

    def forward(self, feature):
        x_pos = func.adaptive_max_pool2d(feature, (self.R, self.R))
        x_all = func.adaptive_max_pool2d(torch.abs(feature), (self.R, self.R))
        mask = x_all > x_pos
        x = x_all * (-(mask == 1).float()) + x_all * (1 - (mask == 1).float())

        x = x.view(x.size(0), -1)
        if self.norm:
            x = func.normalize(x)

        return x
