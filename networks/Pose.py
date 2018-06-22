import setlog
import torch.autograd as auto
import torch.nn as nn
import torch
import networks.Alexnet as Alexnet
import copy
import collections as coll
import torch.nn.functional as func


logger = setlog.get_logger(__name__)


class PoseRegressor(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        input_size = kwargs.pop('input_size', 0)
        size_layer = kwargs.pop('size_layer', 2048)
        num_inter_layers = kwargs.pop('num_inter_layers', 0)
        self.custom_input_size = kwargs.pop('custom_input_size', None)
        self.late_fusion = kwargs.pop('late_fusion', False)

        if kwargs:
            logger.error('Unexpected **kwargs: %r' % kwargs)
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        base_archi = list()

        if num_inter_layers:
            base_archi += [
                ('fc0', nn.Linear(input_size, size_layer)),
                ('relu0',  nn.ReLU(inplace=True)),
                ('do0', nn.Dropout())
            ]
            for layers in self.intermediate_layers(num_inter_layers - 1, size_layer):
                base_archi.append(layers)
            base_archi.append(
                ('pose', nn.Linear(size_layer, 7))
            )
        else:
            base_archi.append(
                ('pose', nn.Linear(input_size, 7))
            )

        self.base_archi = coll.OrderedDict(base_archi)
        self.regressor = nn.Sequential(self.base_archi)
        logger.info('Final regressor architecture:')
        logger.info(self.regressor)

    @staticmethod
    def intermediate_layers(num_layer, size_layer):
        lays = (
            ('fc', nn.Linear(size_layer, size_layer)),
            ('relu', nn.ReLU(inplace=True)),
            ('do', nn.Dropout())
        )
        for i in range(1, num_layer+1):
            for lay in lays:
                yield (lay[0] + str(i), copy.deepcopy(lay[1]))

    def forward(self, x):
        if self.custom_input_size:
            x = func.adaptive_max_pool2d(x, output_size=self.custom_input_size)
        x = x.view(x.size(0), -1)

        if self.late_fusion:
            feats = dict()
            feats['input'] = x
            for name, lay in self.regressor.named_children():
                x = lay(x)
                feats[name] = x
        else:
            x = self.regressor(x)

        p = x[:, 0:3]
        q = x[:, 3:]

        if not self.training:
            q = func.normalize(q)
        if self.late_fusion:
            return {'p': p, 'q': q, 'feats': feats}
        else:
            return {'p': p, 'q': q}


# Poses fusion layers
class Mean(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, p1, p2):
        return {'p': (p1['p'] + p2['p'])/2, 'q': (p1['q'] + p2['q'])/2}


class LateFusion(PoseRegressor):
    def __init__(self, **kwargs):
        self.layer_name = kwargs.pop('layer_name', 'input')
        PoseRegressor.__init__(self, **kwargs)

    def forward(self, p1, p2):
        return PoseRegressor.forward(
            self,
            torch.cat((p1['feats'][self.layer_name], p2['feats'][self.layer_name]), dim=1)
        )


class Main(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        base_archi = kwargs.pop('base_archi', 'Alexnet')
        archi_param = kwargs.pop('archi_param', {
            'load_imagenet': True,
            'batch_norm': False,
            'end_relu': True
        })
        regressor_param = kwargs.pop('regressor_param', {
            'size_layer': 2048,
            'custom_input_size': 3,
            'num_inter_layers': 1
        })
        input_im_size = kwargs.pop('input_im_size', 224)
        self.layers_to_train = kwargs.pop('layers_to_train', 'all')

        if kwargs:
            logger.error('Unexpected **kwargs: %r' % kwargs)
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if base_archi == 'Alexnet':
            self.feature = Alexnet.Feat(**archi_param,
                                        end_max_polling=True)
            size_map = round((input_im_size/self.feature.down_ratio)**2)\
                if not regressor_param.get('custom_input_size') else regressor_param.get('custom_input_size')**2
            input_size = size_map * 256
        else:
            logger.error('No architecture named {}'.format(base_archi))
            raise AttributeError('No architecture named {}'.format(base_archi))

        self.regressor = PoseRegressor(**regressor_param, input_size=input_size)

    def forward(self, x):
        x_feat = self.feature(x)
        pose = self.regressor(x_feat)

        if self.training:
            forward_pass = {'feat': x_feat, **pose}
        else:
            forward_pass = pose

        return forward_pass

    def get_training_layers(self, layers_to_train=None):
        def sub_layers(name):
            return {
                'all': [{'params': self.regressor.parameters()}] + self.feature.get_training_layers('all'),
                'only_feat': self.feature.get_training_layers('all'),
                'only_regressor': [{'params': self.regressor.parameters()}],
                'up_to_conv4': [{'params': self.regressor.parameters()}] +
                               self.feature.get_training_layers('up_to_conv4'),
                'up_to_conv3': [{'params': self.regressor.parameters()}] +
                               self.feature.get_training_layers('up_to_conv3'),
                'up_to_conv2': [{'params': self.regressor.parameters()}] +
                               self.feature.get_training_layers('up_to_conv2'),
                'up_to_conv1': [{'params': self.regressor.parameters()}] +
                               self.feature.get_training_layers('up_to_conv1')
            }.get(name)
        if not layers_to_train:
            layers_to_train = self.layers_to_train
        return sub_layers(layers_to_train)

    def full_save(self):
        return {'feature': self.feature.state_dict(),
                'regressor': self.regressor.state_dict()}


class Deconv(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        base_archi = kwargs.pop('base_archi', 'Alexnet')
        archi_param = kwargs.pop('archi_param', {
            'load_imagenet': True,
            'batch_norm': False,
            'end_relu': True,
            'res': False
        })
        deconv_param = kwargs.pop('deconv_param', {
            'batch_norm': False,
            'res': False
        })
        reg_param = kwargs.pop('reg_param', {
            'size_layer': 2048,
            'custom_input_size': 3,
            'num_inter_layers': 1
        })
        aux_reg_param = kwargs.pop('aux_reg_param', {
            'size_layer': 2048,
            'custom_input_size': 4,
            'num_inter_layers': 1
        })
        fuse_layer =  kwargs.pop('fuse_layer', {'class': 'Mean',
                                                 'param': {}}
        )

        self.auxilary_feat = kwargs.pop('auxilary_feat', 'conv1')
        input_im_size = kwargs.pop('input_im_size', 224)
        self.layers_to_train = kwargs.pop('layers_to_train', 'all')

        if kwargs:
            logger.error('Unexpected **kwargs: %r' % kwargs)
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if base_archi == 'Alexnet':
            self.feature = Alexnet.Feat(**archi_param,
                                        end_max_polling=True,
                                        indices=True,
                                        )
            size_map = round((input_im_size / self.feature.down_ratio))

            input_size = size_map ** 2 * 256 if not reg_param.get('custom_input_size') \
                else reg_param.get('custom_input_size') ** 2 * 256

            self.deconv = Alexnet.Deconv(**deconv_param)
            size_aux_feat = {'conv4': 256, 'relu4': 256,
                             'conv3': 384, 'relu3': 384,
                             'conv2': 192, 'relu2': 192,
                             'conv1': 64, 'relu1': 64}
            aux_size_map = round((size_map / self.deconv.down_ratio[self.auxilary_feat]))
            aux_input_size = aux_size_map ** 2 * size_aux_feat[self.auxilary_feat] \
                if not aux_reg_param.get('custom_input_size') \
                else aux_reg_param.get('custom_input_size') ** 2 * size_aux_feat[self.auxilary_feat]
        else:
            logger.error('No architecture named {}'.format(base_archi))
            raise AttributeError('No architecture named {}'.format(base_archi))

        self.main_reg = PoseRegressor(**reg_param, input_size=input_size)
        self.aux_reg = PoseRegressor(**aux_reg_param, input_size=aux_input_size)
        self.fuse_layer = eval(fuse_layer['class'])(**fuse_layer['param'])

    def forward(self, x):
        x_feat_ouput = self.feature(x)
        if self.feature.res:
            x_deconv_output = self.deconv(x_feat_ouput['output'],
                                          id=x_feat_ouput['id'],
                                          res=x_feat_ouput['res'])
        else:
            x_deconv_output = self.deconv(x_feat_ouput['output'],
                                          id=x_feat_ouput['id'])

        main_pose = self.main_reg(x_feat_ouput['feat'])
        aux_pose = self.aux_reg(x_deconv_output[self.auxilary_feat])
        pose = self.fuse_layer(main_pose, aux_pose)

        if self.training:
            forward_pass = {'main': main_pose,
                            'aux': main_pose,
                            'full': pose,
                            'maps': x_deconv_output['maps']}
        else:
            forward_pass = pose

        return forward_pass

    def get_training_layers(self, layers_to_train=None):
        def sub_layers(name):
            return {
                'all': [{'params': self.main_reg.parameters()},
                        {'params': self.aux_reg.parameters()},
                        {'params': self.deconv.parameters()},
                        {'params': self.fuse_layer.parameters()}
                        ] + self.feature.get_training_layers('all'),
                'only_feat': self.feature.get_training_layers('all'),
                'only_deconv': [{'params': self.main_reg.parameters()},
                                {'params': self.aux_reg.parameters()},
                                {'params': self.fuse_layer.parameters()},
                                {'params': self.deconv.parameters()}],
                'only_regressor': [{'params': self.main_reg.parameters()},
                                   {'params': self.aux_reg.parameters()},
                                   {'params': self.fuse_layer.parameters()}],
                'up_to_conv4': [{'params': self.main_reg.parameters()},
                                {'params': self.aux_reg.parameters()},
                                {'params': self.deconv.parameters()},
                                {'params': self.fuse_layer.parameters()}] +
                               self.feature.get_training_layers('up_to_conv4'),
                'up_to_conv3': [{'params': self.main_reg.parameters()},
                                {'params': self.aux_reg.parameters()},
                                {'params': self.deconv.parameters()},
                                {'params': self.fuse_layer.parameters()}] +
                               self.feature.get_training_layers('up_to_conv3'),
                'up_to_conv2': [{'params': self.main_reg.parameters()},
                                {'params': self.aux_reg.parameters()},
                                {'params': self.deconv.parameters()},
                                {'params': self.fuse_layer.parameters()}] +
                               self.feature.get_training_layers('up_to_conv2'),
                'up_to_conv1': [{'params': self.main_reg.parameters()},
                                {'params': self.aux_reg.parameters()},
                                {'params': self.deconv.parameters()},
                                {'params': self.fuse_layer.parameters()}] +
                               self.feature.get_training_layers('up_to_conv1')
            }.get(name)
        if not layers_to_train:
            layers_to_train = self.layers_to_train
        return sub_layers(layers_to_train)


if __name__ == '__main__':
    tensor_input = torch.rand([10, 3, 224, 224]).cuda()
    '''
    net = Main().cuda()
    feat_output = net(auto.Variable(tensor_input))
    print(feat_output)
    net = Main(
        archi_param={'batch_norm':False, 'end_relu':False}
    ).cuda()
    feat_output = net(auto.Variable(tensor_input.cuda()))
    net.eval()
    feat_output = net(auto.Variable(tensor_input).cuda())
    print(feat_output)
    net.layers_to_train = 'up_to_conv2'
    print(net.get_training_layers())
    net = Main(archi_param={
        'end_relu':True}
    ).eval()
    tensor_input = auto.Variable(torch.ones([1, 3, 224, 224]))
    print(net(tensor_input)['q'])
    net.train()
    print(net(tensor_input)['q'])
    net = Main(archi_param={'batch_norm': True, 'end_relu': True},
               regressor_param={'custom_input_size': 3, 'num_inter_layers': 4}
               )
    print(net(tensor_input)['q'])
    '''
    net = Deconv(
        reg_param={
            'custom_input_size': 3,
            'size_layer': 256,
            'num_inter_layers': 1,
            'late_fusion': True
        },
        aux_reg_param={
            'custom_input_size': 3,
            'size_layer': 256,
            'num_inter_layers': 1,
            'late_fusion': True
    },
        auxilary_feat='conv1',
        fuse_layer={
            'class': 'LateFusion',
            'param': {
                'input_size': 512,
                'layer_name': 'do0'
            }
        }
    ).cuda().eval()
    feat_output = net(auto.Variable(tensor_input))
    print(feat_output)