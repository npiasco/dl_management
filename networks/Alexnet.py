import setlog
import torch.nn as nn
import collections as coll
import torch.autograd as auto
import torch
import torchvision.models as models
import os


logger = setlog.get_logger(__name__)


class Feat(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        batch_norm = kwargs.pop('batch_norm', False)
        end_relu = kwargs.pop('end_relu', False)
        end_max_polling = kwargs.pop('end_max_polling', False)
        load_imagenet = kwargs.pop('load_imagenet', True)
        self.layers_to_train = kwargs.pop('layers_to_train', 'all')

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        base_archi = [
            ('conv0', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),   # 0
            ('relu0', nn.ReLU(inplace=True)),                                   # 1
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2)),                   # 2
            ('conv1', nn.Conv2d(64, 192, kernel_size=5, padding=2)),            # 3
            ('relu1', nn.ReLU(inplace=True)),                                   # 4
            ('pool1', nn.MaxPool2d(kernel_size=3, stride=2)),                   # 5
            ('conv2', nn.Conv2d(192, 384, kernel_size=3, padding=1)),           # 6
            ('relu2', nn.ReLU(inplace=True)),                                   # 7
            ('conv3', nn.Conv2d(384, 256, kernel_size=3, padding=1)),           # 8
            ('relu3', nn.ReLU(inplace=True)),                                   # 9
            ('conv4', nn.Conv2d(256, 256, kernel_size=3, padding=1)),           # 10
            ]

        if batch_norm:
            base_archi.append(('norm4', nn.BatchNorm2d(256)))
            base_archi.insert(9, ('norm3', nn.BatchNorm2d(256)))
            base_archi.insert(7, ('norm2', nn.BatchNorm2d(384)))
            base_archi.insert(4, ('norm1', nn.BatchNorm2d(192)))
            base_archi.insert(1, ('norm0', nn.BatchNorm2d(64)))

        if end_relu:
            base_archi.append(('relu4', nn.ReLU(inplace=True)))

        if end_max_polling:
            base_archi.append(('pool3', nn.MaxPool2d(kernel_size=3, stride=2)))

        self.base_archi = coll.OrderedDict(base_archi)
        self.feature = nn.Sequential(self.base_archi)
        logger.info('Final feature extractor architecture:')
        logger.info(self.feature)
        self._down_ratio = 0

        if load_imagenet:
            self.load()

    def forward(self, x):

        return self.feature(x)

    def load(self):
        alexnet = models.alexnet()
        alexnet = alexnet.features
        alexnet.load_state_dict(torch.load(os.environ['CNN_WEIGHTS'] + 'alexnet_ots.pth'))

        self.base_archi['conv0'].weight.data = alexnet[0].weight.data
        self.base_archi['conv0'].bias.data = alexnet[0].bias.data
        self.base_archi['conv1'].weight.data = alexnet[3].weight.data
        self.base_archi['conv1'].bias.data = alexnet[3].bias.data
        self.base_archi['conv2'].weight.data = alexnet[6].weight.data
        self.base_archi['conv2'].bias.data = alexnet[6].bias.data
        self.base_archi['conv3'].weight.data = alexnet[8].weight.data
        self.base_archi['conv3'].bias.data = alexnet[8].bias.data
        self.base_archi['conv4'].weight.data = alexnet[10].weight.data
        self.base_archi['conv4'].bias.data = alexnet[10].bias.data

    def get_training_layers(self, layers_to_train=None):
        def sub_layers(name):
            return {
                'all': [{'params': self.feature.parameters()}],
                'up_to_conv4': [{'params': layers.parameters()} for layers in
                                list(self.feature.children())[list(self.base_archi.keys()).index('conv4'):]],
                'up_to_conv3': [{'params': layers.parameters()} for layers in
                                list(self.feature.children())[list(self.base_archi.keys()).index('conv3'):]],
                'up_to_conv2': [{'params': layers.parameters()} for layers in
                                list(self.feature.children())[list(self.base_archi.keys()).index('conv2'):]],
                'up_to_conv1': [{'params': layers.parameters()} for layers in
                                list(self.feature.children())[list(self.base_archi.keys()).index('conv1'):]]
            }.get(name)
        if not layers_to_train:
            layers_to_train = self.layers_to_train
        return sub_layers(layers_to_train)

    @property
    def down_ratio(self):
        if 'pool3' in [elem[0] for elem in list(self.feature.named_children())]:
            self._down_ratio = 224 / 6
        else:
            self._down_ratio = 224 / 13
        return self._down_ratio


if __name__ == '__main__':
    net = Feat().cuda()
    tensor_input = torch.rand([10, 3, 224, 224]).cuda()
    feat_output = net(auto.Variable(tensor_input))
    print(feat_output[0])
    net = Feat(batch_norm=False, end_relu=True).cuda()
    feat_output = net(auto.Variable(tensor_input).cuda())
    net.eval()
    feat_output = net(auto.Variable(tensor_input).cuda())
    print(feat_output[0])
    net.layers_to_train = 'up_to_conv2'
    print(net.get_training_layers())
    print(net.down_ratio)
    net = Feat(batch_norm=False, end_relu=True, end_max_polling=True).cuda()
    feat_output = net(auto.Variable(tensor_input).cuda())
    print(feat_output.size())
    print(net.down_ratio)
