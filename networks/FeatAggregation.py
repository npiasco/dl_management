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
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

    def forward(self, x1, x2):

        x = torch.cat((x1, x2), dim=1)

        if self.norm:
            x = func.normalize(x)

        return x
