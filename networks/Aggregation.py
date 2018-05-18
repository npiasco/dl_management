import setlog
import torch.nn.functional as func
import torch.nn as nn
import torch


logger = setlog.get_logger(__name__)


def select_desc(name, params):
    if name == 'RMAC':
        agg = RMAC(**params)
    elif name == 'RAAC':
        agg = RAAC(**params)
    elif name == 'RMean':
        agg = RMean(**params)
    elif name == 'SPOC':
        agg = SPOC(**params)
    elif name == 'Embedding':
        agg = Embedding(**params)
    else:
        raise ValueError('Unknown aggregation method {}'.format(name))

    return agg


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
        agg_params = kwargs.pop('agg_params', {'R': 1, 'norm': True})
        input_size = kwargs.pop('input_size', 256)
        size_feat = kwargs.pop('size_feat', 256)
        self.gate = kwargs.pop('gate', False)
        self.res = kwargs.pop('res', False)

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        self.embed = nn.Conv2d(input_size, size_feat, kernel_size=1)
        self.descriptor = select_desc(agg, agg_params)

        if self.gate:
            self.gatenet = nn.Sequential(
                nn.Conv2d(input_size, size_feat, kernel_size=1),
                nn.Sigmoid()
            )

    def forward(self, feature):
        embeded_feature = self.embed(feature)
        if self.res:
            embeded_feature += feature
        if self.gate:
            gating = self.gatenet(embeded_feature)
            desc = self.descriptor(embeded_feature*gating)
        else:
            desc = self.descriptor(embeded_feature)

        return desc

