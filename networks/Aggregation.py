import setlog
import torch.nn.functional as func
import torch.nn as nn
import torch
import math
import os
import networks.Alexnet as Alexnet
import collections as coll
import networks.CustomLayers as Custom


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
    elif name == 'NetVLAD':
        agg = NetVLAD(**params)
    elif name == 'Encoder':
        agg = nn.Sequential(
            coll.OrderedDict(
                [
                    ('feat', eval(params['base_archi'])(**params['base_archi_param'])),
                    ('agg', select_desc(params['agg'], params['agg_param']))
                ]
            )
        )
    elif name == 'JetEncoder':
        agg = nn.Sequential(
            coll.OrderedDict(
                [
                    ('jet', Custom.IndexEmbedding(size_embedding=3, num_embedding=256, trainable=False, init_jet=True)),
                    ('feat', eval(params['base_archi'])(**params['base_archi_param'])),
                    ('agg', select_desc(params['agg'], params['agg_param']))
                ]
            )
        )

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
        size_feat = kwargs.pop('feat_size', 256)
        self.gate = kwargs.pop('gate', False)
        self.res = kwargs.pop('res', False)
        kernel_size = kwargs.pop('kernel_size', 1)
        stride = kwargs.pop('stride', 1)
        padding = kwargs.pop('padding', 0)
        dilation = kwargs.pop('dilation', 1)

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        self.embed = nn.Conv2d(input_size, size_feat,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation)
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


class NetVLAD(nn.Module):
    """
        Code from Antoine Miech
        @ https://github.com/antoine77340/Mixture-of-Embedding-Experts/blob/master/loupe.py
    """
    def __init__(self, cluster_size, feature_size, **kwargs):
        super().__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        add_batch_norm = kwargs.pop('add_batch_norm', False)
        load = kwargs.pop('load', None)
        alpha = kwargs.pop('alpha', 50)
        trace = kwargs.pop('trace', False)
        self.feat_norm = kwargs.pop('feat_norm', True)
        self.add_bias = kwargs.pop('bias', False)

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        # Reweighting
        self.clusters = nn.Parameter((1 / math.sqrt(feature_size))
                                     * torch.randn(feature_size, cluster_size))

        # Bias
        if self.add_bias:
            self.bias = nn.Parameter((1 / math.sqrt(feature_size))
                                     * torch.randn(cluster_size))

        # Cluster
        self.clusters2 = nn.Parameter((1 / math.sqrt(feature_size))
                                      * torch.randn(1, feature_size, cluster_size))
        if load is not None:
            clusters = torch.load(os.environ['CNN_WEIGHTS'] + load)
            if self.add_bias:
                self.bias.data = -1*alpha*torch.norm(clusters, p=2, dim=1)
            self.clusters2.data = clusters
            self.clusters.data = 2*alpha*clusters.squeeze()
            logger.info('Custom clusters {} have been loaded'.format(os.environ['CNN_WEIGHTS'] + load))
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(cluster_size)
        self.out_dim = cluster_size * feature_size
        self.trace = trace

    def forward(self, x):
        max_sample = x.size(2)*x.size(3)

        if self.feat_norm:
            # Descriptor-wise L2-normalization (see paper)
            x = func.normalize(x)

        x = x.view(x.size(0), self.feature_size, max_sample).transpose(1,2).contiguous()
        x = x.view(-1, self.feature_size)
        assignment = torch.matmul(x, self.clusters) + self.bias if self.add_bias else  torch.matmul(x, self.clusters)
        if self.add_batch_norm:
            assignment = self.batch_norm(assignment)

        assignment = func.softmax(assignment, dim=1)
        assignment = assignment.view(-1, max_sample, self.cluster_size)

        if self.trace:
        # print(torch.max(assignment[0,0]))
            s_tmp = list()
            soft_idx = list()
            for assa in assignment:
                for assa2 in assa:
                    sorted = torch.sort(assa2, descending=True)
                    s_tmp.append(sorted[0][0]/sorted[0][1])
                    soft_idx.append(sorted[1][0])
            print(sum(s_tmp)/len(s_tmp))
            #print(soft_idx)

        a_sum = torch.sum(assignment, -2, keepdim=True)
        a = a_sum * self.clusters2

        assignment = assignment.transpose(1, 2)

        x = x.view(-1, max_sample, self.feature_size)
        vlad = torch.matmul(assignment, x)
        vlad = vlad.transpose(1, 2)
        vlad = vlad - a

        # L2 intra norm
        vlad = func.normalize(vlad)

        # flattening + L2 norm
        vlad = vlad.view(-1, self.cluster_size * self.feature_size)
        vlad = func.normalize(vlad)

        return vlad
