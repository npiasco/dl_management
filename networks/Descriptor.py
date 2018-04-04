import setlog
import networks.Aggregation as Agg
import torch.autograd as auto
import torch.nn as nn
import torch
import networks.Alexnet as Alexnet


logger = setlog.get_logger(__name__)


class Main(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        batch_norm = kwargs.pop('batch_norm', False)
        agg_method = kwargs.pop('agg_method', 'RMAC')
        desc_norm = kwargs.pop('desc_norm', True)
        end_relu = kwargs.pop('end_relu', False)
        base_archi = kwargs.pop('base_archi', 'Alexnet')
        R = kwargs.pop('R', 1)
        load_imagenet = kwargs.pop('load_imagenet', True)
        self.layers_to_train = kwargs.pop('layers_to_train', 'all')

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if base_archi == 'Alexnet':
            self.feature = Alexnet.Feat(batch_norm=batch_norm, end_relu=end_relu, load_imagenet=load_imagenet)

        if agg_method == 'RMAC':
            self.descriptor = Agg.RMAC(R=R, norm=desc_norm)

        logger.info('Descriptor architecture:')
        logger.info(self.descriptor)

    def forward(self, x):
        x_feat = self.feature(x)
        x_desc = self.descriptor(x_feat)

        if self.training:
            forward = {'desc': x_desc, 'feat': x_feat}
        else:
            forward = x_desc

        return forward

    def get_training_layers(self, layers_to_train=None):
        def sub_layers(name):
            return {
                'all': [{'params': self.descriptor.parameters()}] + self.feature.get_training_layers('all'),
                'only_feat': self.feature.get_training_layers('all'),
                'only_descriptor': [{'params': self.descriptor.parameters()}],
                'up_to_conv4': [{'params': self.descriptor.parameters()}] +
                               self.feature.get_training_layers('up_to_conv4'),
                'up_to_conv3': [{'params': self.descriptor.parameters()}] +
                               self.feature.get_training_layers('up_to_conv3'),
                'up_to_conv2': [{'params': self.descriptor.parameters()}] +
                               self.feature.get_training_layers('up_to_conv2'),
                'up_to_conv1': [{'params': self.descriptor.parameters()}] +
                               self.feature.get_training_layers('up_to_conv1')
            }.get(name)
        if not layers_to_train:
            layers_to_train = self.layers_to_train
        return sub_layers(layers_to_train)


if __name__ == '__main__':
    net = Main().cuda()
    tensor_input = torch.rand([10, 3, 224, 224]).cuda()
    feat_output = net(auto.Variable(tensor_input))
    print(feat_output['feat'][0])
    net = Main(batch_norm=False, end_relu=True).cuda()
    feat_output = net(auto.Variable(tensor_input.cuda()))
    net.eval()
    feat_output = net(auto.Variable(tensor_input.cuda()))
    print(feat_output[0])
    net.layers_to_train = 'up_to_conv2'
    print(net.get_training_layers())