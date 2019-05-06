import torch.nn as nn
import torch
import setlog
import matplotlib.pyplot as plt


logger = setlog.get_logger(__name__)


class IndexEmbedding(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        num_embedding = kwargs.pop('num_embedding', 256)
        size_embedding = kwargs.pop('size_embedding', 3)
        self.init_jet = kwargs.pop('init_jet', True)
        self.trainable = kwargs.pop('trainable', False)
        self.amplitude = kwargs.pop('amplitude', 1.0)
        self.min_value = kwargs.pop('min_value', 0.0)

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        self.embedding = nn.Embedding(num_embedding, size_embedding)

        if self.init_jet and size_embedding == 3:
            logger.info('Initialaising embedding weights with JET cmap data...')
            ccmap = plt.get_cmap('jet', lut=num_embedding)
            palet = [ccmap(i)[0:3] for i in range(num_embedding)]
            self.embedding.weight.data = torch.Tensor(palet)

        if not self.trainable:
            self.embedding.weight.requires_grad = False

    def forward(self, feature):
        ori_size = feature.size()

        feature = (feature - self.min_value)/self.amplitude # Index normalization [0,1]
        feature = (feature*(self.embedding.num_embeddings - 1)).long().view(1, -1).detach()

        x = self.embedding(feature)
        x = x.view(ori_size[0], ori_size[2], ori_size[3], -1).transpose(1,3).transpose(2,3).contiguous()

        return x

    def parameters(self):
        if self.trainable:
            return self.embedding.parameters()
        else:
            yield nn.Parameter()

    def named_parameters(self, memo=None, prefix=''):
        if self.trainable:
            return self.embedding.named_parameters(memo=memo, prefix=prefix)
        else:
            yield '', nn.Parameter()
