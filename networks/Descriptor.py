import setlog
import networks.Aggregation as Agg
import torch.autograd as auto
import torch.nn as nn
import torch
import networks.Alexnet as Alexnet
import networks.FeatAggregation as FeatAggregation
import networks.ResNet as Resnet


logger = setlog.get_logger(__name__)


class Main(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        agg_method = kwargs.pop('agg_method', 'RMAC')
        agg_method_param = kwargs.pop('agg_method_param', {'R': 1, 'norm': True})
        base_archi = kwargs.pop('base_archi', 'Alexnet')
        base_archi_param = kwargs.pop('base_archi_param',
                                      {
                                          'load_imagenet': True,
                                          'end_relu': False,
                                          'batch_norm': False
                                      })
        self.layers_to_train = kwargs.pop('layers_to_train', 'all')

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        self.unet = base_archi_param.get('unet', False)

        if base_archi == 'Alexnet':
            self.feature = Alexnet.Feat(**base_archi_param)
        elif base_archi == 'Resnet':
            self.feature = Resnet.Feat(**base_archi_param)
        else:
            raise AttributeError("Unknown base architecture {}".format(base_archi))

        self.descriptor = Agg.select_desc(agg_method, agg_method_param)

        logger.info('Descriptor architecture:')
        logger.info(self.descriptor)

    def forward(self, x):
        x_feat = self.feature(x)

        if self.unet:
            x_desc = self.descriptor(x_feat['feat'])
        else:
            x_desc = self.descriptor(x_feat)

        if self.training:
            if self.unet:
                forward_pass = {'desc': x_desc,
                                'feat': x_feat['feat'],
                                'res_1': x_feat['res_1'],
                                'res_2': x_feat['res_2']}
            else:
                forward_pass = {'desc': x_desc, 'feat': x_feat}
        else:
            if self.unet:
                forward_pass = {'desc': x_desc,
                                'feat': x_feat['feat'],
                                'res_1': x_feat['res_1'],
                                'res_2': x_feat['res_2']}
            else:
                forward_pass = x_desc

        return forward_pass

    def get_training_layers(self, layers_to_train=None):
        if not layers_to_train:
            layers_to_train = self.layers_to_train

        if layers_to_train == 'all':
            train_parameters = [{'params': self.descriptor.parameters()}] + self.feature.get_training_layers('all')
        elif layers_to_train == 'only_feat':
            train_parameters = self.feature.get_training_layers('all')
        elif layers_to_train == 'only_descriptor':
            train_parameters = [{'params': self.descriptor.parameters()}]
        elif layers_to_train == 'up_to_conv4':
            train_parameters = [{'params': self.descriptor.parameters()}] + \
                               self.feature.get_training_layers('up_to_conv4')
        elif layers_to_train == 'up_to_conv3':
            train_parameters = [{'params': self.descriptor.parameters()}] +\
                               self.feature.get_training_layers('up_to_conv3')
        elif layers_to_train == 'up_to_conv2':
            train_parameters = [{'params': self.descriptor.parameters()}] +\
                               self.feature.get_training_layers('up_to_conv2')
        elif layers_to_train == 'up_to_conv1':
            train_parameters = [{'params': self.descriptor.parameters()}] +\
                               self.feature.get_training_layers('up_to_conv1')
        elif layers_to_train == 'only_jet':
            train_parameters = self.feature.get_training_layers('only_jet')
        elif layers_to_train == 'no_layer':
            train_parameters = []
        else:
            raise KeyError('No behaviour for layers_to_train = {}'.format(layers_to_train))

        return train_parameters

    def full_save(self, discard_tf=False):
        if discard_tf:
            del self.feature.base_archi['jet_tf']
            self.feature.feature = nn.Sequential(self.feature.base_archi)
        return {'feature': self.feature.state_dict(),
                'descriptor': self.descriptor.state_dict()}


class Deconv(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        agg_method = kwargs.pop('agg_method', 'RMAC')
        agg_method_param = kwargs.pop('agg_method_param', {'R': 1, 'norm': True})
        aux_agg = kwargs.pop('aux_agg', 'RMAC')
        aux_agg_param = kwargs.pop('aux_agg_param', {'R': 1, 'norm': True})
        feat_agg_method = kwargs.pop('feat_agg_method', 'Concat')
        feat_agg_params = kwargs.pop('feat_agg_param', dict())
        self.auxilary_feat = kwargs.pop('auxilary_feat', 'conv1')
        enc_base_archi = kwargs.pop('enc_base_archi', 'Alexnet')
        enc_base_param = kwargs.pop('enc_base_param',
                                    {'batch_norm': False,
                                     'end_relu': False,
                                     'load_imagenet': True
                                     })
        dec_base_archi = kwargs.pop('dec_base_archi', 'Alexnet')
        dec_base_param = kwargs.pop('dec_base_param',
                                    {'batch_norm': False,
                                     })

        self.res = kwargs.pop('res', False)
        self.unet = kwargs.pop('unet', False)
        self.layers_to_train = kwargs.pop('layers_to_train', 'all')
        self.return_all_desc = kwargs.pop('return_all_desc', False)
        self.return_maps = kwargs.pop('return_maps', False)

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if enc_base_archi == 'Alexnet':
            self.feature = Alexnet.Feat(**enc_base_param,
                                        res=self.res,
                                        unet=self.unet,
                                        indices=True,
                                        end_max_polling=True)
        else:
            raise AttributeError("Unknown base architecture {}".format(enc_base_archi))

        if dec_base_archi == 'Alexnet':
            self.deconv = Alexnet.Deconv(**dec_base_param,
                                         res=self.res,
                                         unet=self.unet)
        else:
            raise AttributeError("Unknown base architecture {}".format(dec_base_archi))

        self.descriptor = Agg.select_desc(agg_method, agg_method_param)
        self.aux_descriptor = Agg.select_desc(aux_agg, aux_agg_param)

        if feat_agg_method == 'Concat':
            self.feat_agg = FeatAggregation.Concat(**feat_agg_params)
        elif feat_agg_method == 'Sum':
            self.feat_agg = FeatAggregation.Sum(**feat_agg_params)
        else:
            raise AttributeError("Unknown feat aggregation method {}".format(feat_agg_method))

        logger.info('Descriptor architecture:')
        logger.info('Main:\n{}'.format(self.descriptor))
        logger.info('Aux:\n{}'.format(self.aux_descriptor))
        logger.info('Fuse:\n{}'.format(self.feat_agg))

    def forward(self, x):
        x_feat_ouput = self.feature(x)
        if self.unet:
            x_deconv_output = self.deconv(x_feat_ouput['output'],
                                          id=x_feat_ouput['id'],
                                          res=x_feat_ouput['res'])
        elif self.res:
            x_deconv_output = self.deconv(x_feat_ouput['output'],
                                          *x_feat_ouput['res'])
        else:
            x_deconv_output = self.deconv(x_feat_ouput['output'],
                                          id=x_feat_ouput['id'])

        x_desc = self.feat_agg(
            self.descriptor(x_feat_ouput['feat']),
            self.aux_descriptor(x_deconv_output[self.auxilary_feat])
        )

        if self.training:
            if self.return_all_desc:
                forward_pass = {
                    'desc':
                        {
                            'main': self.descriptor(x_feat_ouput['feat']),
                            'aux': self.aux_descriptor(x_deconv_output[self.auxilary_feat]),
                            'full': x_desc,
                        },
                    'maps': x_deconv_output['maps']
                }
            else:
                forward_pass = {'desc': x_desc, 'maps': x_deconv_output['maps']}
            forward_pass['feat'] = {
                'main': x_feat_ouput['feat'],
                'aux': x_deconv_output[self.auxilary_feat]
            }
        elif self.return_maps:
            forward_pass = {
                'desc': x_desc,
                'maps': x_deconv_output['maps']
            }
        else:
            forward_pass = x_desc

        return forward_pass

    def get_training_layers(self, layers_to_train=None):
        def sub_layers(name):
            return {
                'all': [{'params': self.descriptor.parameters()},
                        {'params': self.aux_descriptor.parameters()},
                        {'params': self.deconv.parameters()},
                        {'params': self.feat_agg.parameters()}] + self.feature.get_training_layers('all'),
                'only_feat': self.feature.get_training_layers('all'),
                'only_descriptor': [{'params': self.descriptor.parameters()},
                                    {'params': self.aux_descriptor.parameters()},
                                    {'params': self.feat_agg.parameters()}],
                'only_deconv': [{'params': self.descriptor.parameters()},
                                {'params': self.aux_descriptor.parameters()},
                                {'params': self.feat_agg.parameters()},
                                {'params': self.deconv.parameters()}],
                'up_to_conv4': [{'params': self.descriptor.parameters()},
                                {'params': self.aux_descriptor.parameters()},
                                {'params': self.deconv.parameters()},
                                {'params': self.feat_agg.parameters()}
                                ] + self.feature.get_training_layers('up_to_conv4'),
                'up_to_conv3': [{'params': self.descriptor.parameters()},
                                {'params': self.aux_descriptor.parameters()},
                                {'params': self.deconv.parameters()},
                                {'params': self.feat_agg.parameters()}
                                ] + self.feature.get_training_layers('up_to_conv3'),
                'up_to_conv2': [{'params': self.descriptor.parameters()},
                                {'params': self.aux_descriptor.parameters()},
                                {'params': self.deconv.parameters()},
                                {'params': self.feat_agg.parameters()}
                                ] + self.feature.get_training_layers('up_to_conv2'),
                'up_to_conv1': [{'params': self.descriptor.parameters()},
                                {'params': self.aux_descriptor.parameters()},
                                {'params': self.deconv.parameters()},
                                {'params': self.feat_agg.parameters()}
                                ] + self.feature.get_training_layers('up_to_conv1')
            }.get(name)
        if not layers_to_train:
            layers_to_train = self.layers_to_train
        return sub_layers(layers_to_train)

    def full_save(self, discard_tf=False):
        if discard_tf:
            del self.feature.base_archi['jet_tf']
            self.feature.feature = nn.Sequential(self.feature.base_archi)

        return {'feature': self.feature.state_dict(),
                'deconv': self.deconv.state_dict(),
                'descriptor': self.descriptor.state_dict(),
                'aux_descriptor': self.aux_descriptor.state_dict(),
                'feature_agg': self.feat_agg.state_dict()}


if __name__ == '__main__':
    tensor_input = torch.rand([10, 3, 224, 224]).cuda()
    """
    net = Main().cuda()
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
    """

    tensor_input = torch.rand([5, 1, 224, 224]).cuda()

    net = Main(
        base_archi='Resnet',
        base_archi_param={
            'num_layer': 50,
            'jet_tf': True,
            'jet_tf_is_trainable': False
        }
    ).cuda()
    feat_output = net(auto.Variable(tensor_input))
    print(feat_output['desc'].size())
    #print(net.get_training_layers('only_jet'))
    #net.full_save(discard_tf=True)

    """
    net = Deconv(agg_method='RMAC',
                 res=False,
                 unet=True,
                 auxilary_feat='maps',
                 feat_agg_method='Concat',
                 aux_agg='Encoder',
                 aux_agg_param={
                     'base_archi': 'Alexnet.Feat',
                     'base_archi_param': {
                         'mono': False,
                         'jet_tf': True
                     },
                     'agg': 'RMAC',
                     'agg_param': {}
                 },
                 dec_base_param={
                     'upsample': True,
                     'final_jet_tf': False
                 },
                 return_all_desc=True,
                 ).cuda()

    feat_output = net(auto.Variable(tensor_input))
    print(feat_output['desc'])
    print(feat_output['maps'].size())
    print(net.get_training_layers('all'))
    """