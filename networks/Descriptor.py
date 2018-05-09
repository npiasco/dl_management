import setlog
import networks.Aggregation as Agg
import torch.autograd as auto
import torch.nn as nn
import torch
import networks.Alexnet as Alexnet
import networks.FeatAggregation as FeatAggregation

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
            self.feature = Alexnet.Feat(batch_norm=batch_norm,
                                        end_relu=end_relu,
                                        load_imagenet=load_imagenet)
        else:
            raise AttributeError("Unknown base architecture {}".format(base_archi))

        if agg_method == 'RMAC':
            self.descriptor = Agg.RMAC(R=R, norm=desc_norm)
        elif agg_method == 'RAAC':
            self.descriptor = Agg.RAAC(R=R, norm=desc_norm)
        elif agg_method == 'RMean':
            self.descriptor = Agg.RMean(R=R, norm=desc_norm)
        elif agg_method == 'SPOC':
            self.descriptor = Agg.SPOC(norm=desc_norm)
        else:
            raise AttributeError("Unknown aggregation method {}".format(agg_method))

        logger.info('Descriptor architecture:')
        logger.info(self.descriptor)

    def forward(self, x):
        x_feat = self.feature(x)
        x_desc = self.descriptor(x_feat)

        if self.training:
            forward_pass = {'desc': x_desc, 'feat': x_feat}
        else:
            forward_pass = x_desc

        return forward_pass

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

    def full_save(self):
        return {'feature': self.feature.state_dict(),
                'descriptor': self.descriptor.state_dict()}


class Deconv(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        batch_norm = kwargs.pop('batch_norm', False)
        agg_method = kwargs.pop('agg_method', 'RMAC')
        feat_agg_method = kwargs.pop('feat_agg_method', 'Concat')
        feat_agg_params = kwargs.pop('feat_agg_params', dict())
        self.auxilary_feat = kwargs.pop('auxilary_feat', 'conv1')
        desc_norm = kwargs.pop('desc_norm', True)
        self.end_relu = kwargs.pop('end_relu', False)
        base_archi = kwargs.pop('base_archi', 'Alexnet')
        R = kwargs.pop('R', 1)
        load_imagenet = kwargs.pop('load_imagenet', True)
        self.res = kwargs.pop('res', False)
        self.layers_to_train = kwargs.pop('layers_to_train', 'all')

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if base_archi == 'Alexnet':
            self.feature = Alexnet.Feat(batch_norm=batch_norm,
                                        end_relu=self.end_relu,
                                        load_imagenet=load_imagenet,
                                        res=self.res,
                                        indices=True)
            self.deconv = Alexnet.Deconv(batch_norm=batch_norm,
                                         res=self.res)
        else:
            raise AttributeError("Unknown base architecture {}".format(base_archi))

        if agg_method == 'RMAC':
            self.descriptor = Agg.RMAC(R=R, norm=desc_norm)
        elif agg_method == 'RAAC':
            self.descriptor = Agg.RAAC(R=R, norm=desc_norm)
        elif agg_method == 'RMean':
            self.descriptor = Agg.RMean(R=R, norm=desc_norm)
        elif agg_method == 'SPOC':
            self.descriptor = Agg.SPOC(norm=desc_norm)
        else:
            raise AttributeError("Unknown aggregation method {}".format(agg_method))

        if feat_agg_method == 'Concat':
            self.feat_agg = FeatAggregation.Concat(**feat_agg_params)
        else:
            raise AttributeError("Unknown feat aggregation method {}".format(feat_agg_method))

        logger.info('Descriptor architecture:')
        logger.info(self.descriptor)

    def forward(self, x):
        x_feat_ouput = self.feature(x)
        if self.res:
            x_deconv_output = self.deconv(x_feat_ouput['output'],
                                          id=x_feat_ouput['id'],
                                          res=x_feat_ouput['res'])
        else:
            x_deconv_output = self.deconv(x_feat_ouput['output'],
                                          id=x_feat_ouput['id'])

        x_desc = self.feat_agg(
            self.descriptor(x_feat_ouput['feat']),
            self.descriptor(x_deconv_output[self.auxilary_feat])
        )

        if self.training:
            forward_pass = {'desc': x_desc, 'maps': x_deconv_output['maps']}
        else:
            forward_pass = x_desc

        return forward_pass

    def get_training_layers(self, layers_to_train=None):
        def sub_layers(name):
            return {
                'all': [{'params': self.descriptor.parameters()},
                        {'params': self.deconv.parameters()},
                        {'params': self.feat_agg.parameters()}] + self.feature.get_training_layers('all'),
                'only_feat': self.feature.get_training_layers('all'),
                'only_descriptor': [{'params': self.descriptor.parameters()},
                                    {'params': self.feat_agg.parameters()}],
                'only_deconv': [{'params': self.descriptor.parameters()},
                                {'params': self.feat_agg.parameters()},
                                {'params': self.deconv.parameters()}],
                'up_to_conv4': [{'params': self.descriptor.parameters()},
                                {'params': self.deconv.parameters()},
                                {'params': self.feat_agg.parameters()}
                                ] + self.feature.get_training_layers('up_to_conv4'),
                'up_to_conv3': [{'params': self.descriptor.parameters()},
                                {'params': self.deconv.parameters()},
                                {'params': self.feat_agg.parameters()}
                                ] + self.feature.get_training_layers('up_to_conv3'),
                'up_to_conv2': [{'params': self.descriptor.parameters()},
                                {'params': self.deconv.parameters()},
                                {'params': self.feat_agg.parameters()}
                                ] + self.feature.get_training_layers('up_to_conv2'),
                'up_to_conv1': [{'params': self.descriptor.parameters()},
                                {'params': self.deconv.parameters()},
                                {'params': self.feat_agg.parameters()}
                                ] + self.feature.get_training_layers('up_to_conv1')
            }.get(name)
        if not layers_to_train:
            layers_to_train = self.layers_to_train
        return sub_layers(layers_to_train)

    def full_save(self):
        return {'feature': self.feature.state_dict(),
                'deconv': self.deconv.state_dict(),
                'descriptor': self.descriptor.state_dict(),
                'feature_agg': self.feat_agg.state_dict()}


if __name__ == '__main__':
    """
    net = Main().cuda()
    tensor_input = torch.rand([10, 3, 224, 224]).cuda()
    feat_output = net(auto.Variable(tensor_input))
    print(feat_output['desc'][0])
    net = Main(batch_norm=False, end_relu=False).cuda()
    feat_output = net(auto.Variable(tensor_input.cuda()))
    net.eval()
    feat_output = net(auto.Variable(tensor_input).cuda())
    print(feat_output[0])
    net.layers_to_train = 'up_to_conv2'
    print(net.get_training_layers())
    net = Main(end_relu=True).eval()
    tensor_input = auto.Variable(torch.ones([1, 3, 224, 224]))
    print(net(tensor_input))

    net = Main(agg_method='RMean', end_relu=True, desc_norm=False).cuda()
    tensor_input = torch.rand([1, 3, 224, 224]).cuda()
    feat_output = net(auto.Variable(tensor_input))
    print(feat_output['desc'])
    net = Main(agg_method='SPOC', end_relu=True, desc_norm=False).cuda()
    feat_output = net(auto.Variable(tensor_input))
    print(feat_output['desc'])
    """
    tensor_input = torch.rand([10, 3, 224, 224]).cuda()
    net = Deconv(agg_method='RMAC',
                 end_relu=False,
                 res=True,
                 auxilary_feat='conv1',
                 batch_norm=True,
                 feat_agg_method='Concat',
                 R=2
                 ).cuda()

    feat_output = net(auto.Variable(tensor_input))
    print(feat_output['desc'])
    print(net.get_training_layers('all'))
    print(net.get_training_layers('only_deconv'))