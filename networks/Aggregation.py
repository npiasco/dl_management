import setlog
import torch.nn.functional as func
import torch.nn as nn
import torch


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


class RMean(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.R = kwargs.pop('R', 1)  # R=1, Global Max pooling
        self.norm = kwargs.pop('norm', True)
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

    def forward(self, feature):
        x = func.adaptive_avg_pool2d(feature, (self.R, self.R))
        x = x.view(x.size(0), -1)
        if self.norm:
            x = func.normalize(x)

        return x


class SPOC(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.norm = kwargs.pop('norm', True)
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

    def forward(self, feature):
        x = torch.sum(torch.sum(feature, dim=-1), dim=-1)
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


class Embedding(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        agg = kwargs.pop('agg', 'RMAC')
        R = kwargs.pop('R', 1)
        size = kwargs.pop('size', 256)
        norm = kwargs.pop('norm', True)
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        self.embeds = nn.Conv2d(size, size, [1, 1])

        if agg == 'RMAC':
            self.descriptor = RMAC(R=R, norm=norm)
        elif agg == 'RAAC':
            self.descriptor = RAAC(R=R, norm=norm)
        elif agg == 'RMean':
            self.descriptor = RMean(R=R, norm=norm)
        elif agg == 'SPOC':
            self.descriptor = SPOC(norm=norm)
        else:
            raise AttributeError("Unknown aggregation method {}".format(agg))

    def forward(self, feature):
        feature = self.embeds(feature)
        desc = self.descriptor(feature)

        return desc

