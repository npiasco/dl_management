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
        trainable = kwargs.pop('trainable', True)

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        self.embedding = nn.Embedding(num_embedding, size_embedding)

        if self.init_jet:
            ccmap = plt.get_cmap('jet', lut=num_embedding)
            palet = [ccmap(i)[0:3] for i in range(num_embedding)]
            self.embedding.weight.data = torch.Tensor(palet)

        if not trainable:
            self.embedding.weight.requires_grad = False

    def forward(self, feature):

        ori_size = feature.size()
        feature = feature.mul(self.embedding.num_embeddings).long().view(1, -1).detach()

        if self.init_jet:
            feature -= torch.min(feature)

        x = self.embedding(feature)

        x = x.view(ori_size[0], ori_size[2], ori_size[3], -1).transpose(1,3).contiguous()
        return x
