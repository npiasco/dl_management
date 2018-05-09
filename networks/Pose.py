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
        x = self.regressor(x)
        p = x[:, 0:3]
        q = x[:, 3:]

        if not self.training:
            q = func.normalize(q)

        return {'p': p, 'q': q}


class Main(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        batch_norm = kwargs.pop('batch_norm', False)
        end_relu = kwargs.pop('end_relu', True)
        base_archi = kwargs.pop('base_archi', 'Alexnet')
        load_imagenet = kwargs.pop('load_imagenet', True)

        input_im_size = kwargs.pop('input_im_size', 224)
        self.layers_to_train = kwargs.pop('layers_to_train', 'all')
        size_reg_layer = kwargs.pop('size_layer', 2048)
        num_inter_reg_layers = kwargs.pop('num_inter_layers', 1)
        custom_input_size = kwargs.pop('custom_input_size', 3)

        if kwargs:
            logger.error('Unexpected **kwargs: %r' % kwargs)
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if base_archi == 'Alexnet':
            self.feature = Alexnet.Feat(batch_norm=batch_norm,
                                        end_relu=end_relu,
                                        load_imagenet=load_imagenet,
                                        end_max_polling=True)
            size_map = round((input_im_size/self.feature.down_ratio)**2) if not custom_input_size else \
                custom_input_size**2
            input_size = size_map * 256

        else:
            logger.error('No architecture named {}'.format(base_archi))
            raise AttributeError('No architecture named {}'.format(base_archi))

        self.regressor = PoseRegressor(size_layer=size_reg_layer,
                                       num_inter_layers=num_inter_reg_layers,
                                       input_size=input_size,
                                       custom_input_size=custom_input_size)

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



if __name__ == '__main__':
    net = Main().cuda()
    tensor_input = torch.rand([10, 3, 224, 224]).cuda()
    feat_output = net(auto.Variable(tensor_input))
    print(feat_output)
    net = Main(batch_norm=False, end_relu=False).cuda()
    feat_output = net(auto.Variable(tensor_input.cuda()))
    net.eval()
    feat_output = net(auto.Variable(tensor_input).cuda())
    print(feat_output)
    net.layers_to_train = 'up_to_conv2'
    print(net.get_training_layers())
    net = Main(end_relu=True).eval()
    tensor_input = auto.Variable(torch.ones([1, 3, 224, 224]))
    print(net(tensor_input)['q'])
    net.train()
    print(net(tensor_input)['q'])
    net = Main(custom_input_size=3, batch_norm=True, end_relu=True, num_inter_layers=4)
    print(net(tensor_input)['q'])
