import torch.nn as nn
import torch.nn.functional as func
import torch
import setlog
import networks.Aggregation as Agg


logger = setlog.get_logger(__name__)


class GatedFuse(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        size = kwargs.pop('size', 256)
        self.norm = kwargs.pop('norm', False)
        self.gate_type = kwargs.pop('gate_type', 'linear')
        self.cat_type = kwargs.pop('cat_type', 'cat')
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)
        if self.cat_type not in ('cat', 'sum'):
            raise AttributeError('No cat_type named {}'.format(self.cat_type))
        if self.gate_type not in ('linear', ):
            raise AttributeError('No self.gate_type named {}'.format(self.gate_type))

        if self.gate_type == 'linear':
            self.gate_x1 = nn.Sequential(
                nn.Linear(2*size, 1),
                nn.Sigmoid()
            )
            self.gate_x2 = nn.Sequential(
                nn.Linear(2*size, 1),
                nn.Sigmoid()
            )

    def get_training_layers(self, layers_to_train=None):
        return [{'params': self.parameters()},]

    def forward(self, x1, x2):
        x_cat = torch.cat((x1, x2), dim=1)

        g1 = self.gate_x1(x_cat)
        g2 = self.gate_x2(x_cat)

        if self.cat_type == 'sum':
            x = g1*x1 + g2*x2
        elif self.cat_type == 'cat':
            x = torch.cat((g1*x1, g2*x2), dim=1)
        else:
            raise AttributeError('No cat_type named {}'.format(self.cat_type))
        if self.norm:
            x = func.normalize(x)

        return x


class ClustersReweight(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        self.size = kwargs.pop('size', 256)
        self.n_clusters = kwargs.pop('n_clusters', 64)
        self.norm = kwargs.pop('norm', False)

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        self.gate_x1 = nn.Sequential(
            nn.Linear(2 * self.size * self.n_clusters, self.n_clusters),
            nn.Sigmoid()
        )
        self.gate_x2 = nn.Sequential(
            nn.Linear(2 * self.size * self.n_clusters, self.n_clusters),
            nn.Sigmoid()
        )

    def get_training_layers(self, layers_to_train=None):
        return [{'params': self.parameters()},]

    def forward(self, x1, x2):
        x_cat = torch.cat((x1, x2), dim=1)

        n_b = x1.size(0)

        g1 = self.gate_x1(x_cat).unsqueeze(-1).expand(-1, -1, self.size).contiguous().view(n_b, -1)
        g2 = self.gate_x2(x_cat).unsqueeze(-1).expand(-1, -1, self.size).contiguous().view(n_b, -1)

        x = torch.cat((g1*x1, g2*x2), dim=1)

        if self.norm:
            x = func.normalize(x)

        return x


class Concat(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        self.norm = kwargs.pop('norm', True)
        self.norm_x1 = kwargs.pop('norm_x1', False)
        self.norm_x2 = kwargs.pop('norm_x2', False)
        self.main_ratio = kwargs.pop('main_ratio', 1)
        self.aux_ratio = kwargs.pop('aux_ratio', 1)
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

    def forward(self, x1, x2):

        if self.norm_x1:
            x1 = func.normalize(x1)
        if self.norm_x2:
            x2 = func.normalize(x2)

        x = torch.cat((x1*self.main_ratio,
                       x2*self.aux_ratio), dim=1)

        if self.norm:
            x = func.normalize(x)

        return x

    def get_training_layers(self, layers_to_train=None):
        return []

    def full_save(self, discard_tf=False):
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

class FuseVLAD(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        self.main_ratio = kwargs.pop('main_ratio', 1)
        self.aux_ratio = kwargs.pop('aux_ratio', 1)
        vlad_param = kwargs.pop('vlad_param', {})
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        self.vlad = Agg.NetVLAD(**vlad_param)

    def forward(self, x1, x2):

        x = torch.cat((x1 * self.main_ratio,
                       x2 * self.aux_ratio), dim=1)

        x = self.vlad(x)

        return x

    def get_training_layers(self, layers_to_train=None):
        return [{'params': self.vlad.parameters()}]

    def full_save(self):
        return {'vlad': self.vlad.state_dict()}