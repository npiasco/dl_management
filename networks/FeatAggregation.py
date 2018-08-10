import torch.nn as nn
import torch.nn.functional as func
import torch
import setlog


logger = setlog.get_logger(__name__)


class GatedFuse(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        size = kwargs.pop('size', 256)
        self.norm = kwargs.pop('norm', True)
        self.gate_type = kwargs.pop('gate_type', 'linear')
        self.cat_type = kwargs.pop('cat_type', 'cat')
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)
        if self.cat_type not in ('cat', 'sum'):
            raise AttributeError('No cat_type named {}'.format(self.cat_type))
        if self.gate_type not in ('linear',):
            raise AttributeError('No self.gate_type named {}'.format(self.gate_type))

        if self.gate_type == 'linear':
            self.gate = nn.Sequential(
                nn.Linear(size*2, size),
                nn.Sigmoid()
            )

    def get_training_layers(self):
        return [{'params': self.gate.parameters()}]


    def forward(self, x1, x2):
        x_cat = torch.cat((x1, x2), dim=1)

        g = self.gate(x_cat)

        if self.cat_type == 'sum':
            x = g*x1 + (1-g)*x2
        elif self.cat_type == 'cat':
            x = torch.cat((g*x1, (1-g)*x2), dim=1)
        else:
            raise AttributeError('No cat_type named {}'.format(self.cat_type))
        if self.norm:
            x = func.normalize(x)

        return x


class Concat(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        self.norm = kwargs.pop('norm', True)
        self.main_ratio = kwargs.pop('main_ratio', 1)
        self.aux_ratio = kwargs.pop('aux_ratio', 1)
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

    def forward(self, x1, x2):

        x = torch.cat((x1*self.main_ratio,
                       x2*self.aux_ratio), dim=1)

        if self.norm:
            x = func.normalize(x)

        return x

    def get_training_layers(self, layers_to_train=None):
        return []

    def full_save(self):
        return {}


class Sum(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        self.norm = kwargs.pop('norm', True)
        self.main_ratio = kwargs.pop('main_ratio', 1.0)
        self.aux_ratio = kwargs.pop('aux_ratio', 1.0)
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

    def forward(self, x1, x2):

        x = x1*self.main_ratio + x2*self.aux_ratio

        if self.norm:
            x = func.normalize(x)

        return x

    def get_training_layers(self):
        return []