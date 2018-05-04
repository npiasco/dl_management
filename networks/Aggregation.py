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
        x, ind = func.adaptive_max_pool2d(torch.abs(feature), (self.R, self.R), return_indices=True)

        feature = feature.view(feature.size(0), feature.size(1), -1)
        max_values = auto.Variable(torch.zeros(x.size())).cuda() if feature.is_cuda \
            else auto.Variable(torch.zeros(x.size()))
        ind = ind.view(ind.size(0), -1).cpu().data.numpy()
        for i, indices in enumerate(ind):
            for j, indice in enumerate(indices):
                max_values[i, j] = feature[i, j, indice].data

        x = x * torch.sign(max_values)

        x = x.view(x.size(0), -1)
        if self.norm:
            x = func.normalize(x)

        return x