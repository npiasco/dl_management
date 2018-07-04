import setlog
import torch.nn as nn
import collections as coll
import torch.autograd as auto
import torch.nn.functional as func
import torch
import torchvision.models as models
import os
import matplotlib.pyplot as plt


logger = setlog.get_logger(__name__)


class Feat(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        batch_norm = kwargs.pop('batch_norm', False)
        end_relu = kwargs.pop('end_relu', False)
        end_max_polling = kwargs.pop('end_max_polling', False)
        load_imagenet = kwargs.pop('load_imagenet', True)
        self.layers_to_train = kwargs.pop('layers_to_train', 'all')
        self.indices = kwargs.pop('indices', False)
        self.res = kwargs.pop('res', False)
        self.unet = kwargs.pop('unet', False)
        mono = kwargs.pop('mono', False)
        self.jet_tf = plt.get_cmap('jet', lut=1024) if kwargs.pop('jet_tf', False) else False

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        i_channel = 1 if mono else 3

        base_archi = [
            ('conv0', nn.Conv2d(i_channel, 64, kernel_size=11, stride=4, padding=2)),   # 0
            ('relu0', nn.ReLU(inplace=True)),                                   # 1
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, return_indices=self.indices)),                   # 2
            ('conv1', nn.Conv2d(64, 192, kernel_size=5, padding=2)),            # 3
            ('relu1', nn.ReLU(inplace=True)),                                   # 4
            ('pool1', nn.MaxPool2d(kernel_size=3, stride=2, return_indices=self.indices)),                   # 5
            ('conv2', nn.Conv2d(192, 384, kernel_size=3, padding=1)),           # 6
            ('relu2', nn.ReLU(inplace=True)),                                   # 7
            ('conv3', nn.Conv2d(384, 256, kernel_size=3, padding=1)),           # 8
            ('relu3', nn.ReLU(inplace=True)),                                   # 9
            ('conv4', nn.Conv2d(256, 256, kernel_size=3, padding=1)),           # 10
            ]

        if batch_norm:
            base_archi.insert(9, ('norm3', nn.BatchNorm2d(256)))
            base_archi.insert(7, ('norm2', nn.BatchNorm2d(384)))
            base_archi.insert(4, ('norm1', nn.BatchNorm2d(192)))
            base_archi.insert(1, ('norm0', nn.BatchNorm2d(64)))

        if end_relu:
            base_archi.append(('relu4', nn.ReLU(inplace=True)))

        if end_max_polling:
            base_archi.append(('pool3', nn.MaxPool2d(kernel_size=3, stride=2, return_indices=self.indices)))

        self.base_archi = coll.OrderedDict(base_archi)
        self.feature = nn.Sequential(self.base_archi)
        logger.info('Final feature extractor architecture:')
        logger.info(self.feature)
        self._down_ratio = 0

        if load_imagenet:
            self.load(mono)

    def forward(self, x):
        if self.jet_tf:
            # TODO: change into pytorch pipeline
            f_cuda = lambda var : var.cuda() if x.is_cuda else var
            #print(torch.Tensor(self.jet_tf(x.squeeze().cpu().data.numpy()).transpose((0,3,1,2))).size())
            x = x - torch.min(x)
            new_x = torch.autograd.Variable(
                torch.Tensor(self.jet_tf(x.squeeze(1).cpu().data.numpy()).transpose((0, 3, 1, 2)))[:,0:3,:,:]
            )
            x = f_cuda(new_x)

        if self.indices or self.res:
            ind = list()
            res = list()
            for name, lay in self.feature.named_children():
                if 'pool' in name:
                    x, i = lay(x)
                    ind.append(i)
                else:
                    x = lay(x)
                if self.res and name in ['pool0', 'pool1']:
                    res.append(x)
                if self.unet and name in ['relu0', 'relu1']:
                    res.append(x)
                if name == 'conv4':
                    feat = x

            return {'output': x, 'feat': feat, 'id': ind, 'res': res}
        else:
            return self.feature(x)

    def load(self, mono):
        logger.info('Loading pretrained weight (mono = {})'.format(mono))
        alexnet = models.alexnet()
        alexnet = alexnet.features
        alexnet.load_state_dict(torch.load(os.environ['CNN_WEIGHTS'] + 'alexnet_ots.pth'))

        if not mono:
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


class Deconv(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        batch_norm = kwargs.pop('batch_norm', False)
        end_relu = kwargs.pop('end_relu', False)
        self.layers_to_train = kwargs.pop('layers_to_train', 'all')
        self.res = kwargs.pop('res', False)
        self.unet = kwargs.pop('unet', False)

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        unet_multp = 2 if self.unet else 1

        base_archi = [
            ('unpool4', nn.MaxUnpool2d(kernel_size=3, stride=2)),       # 0
            ('conv4', nn.Conv2d(256, 256, kernel_size=3, padding=1)),   # 1
            ('relu4', nn.ReLU(inplace=True)),                           # 2
            ('conv3', nn.Conv2d(256, 384, kernel_size=3, padding=1)),   # 3
            ('relu3', nn.ReLU(inplace=True)),                           # 4
            ('conv2', nn.Conv2d(384, 192, kernel_size=3, padding=1)),   # 5
            ('relu2', nn.ReLU(inplace=True)),                           # 6
            ('unpool2', nn.MaxUnpool2d(kernel_size=3, stride=2)),       # 7
            ('conv1', nn.Conv2d(unet_multp * 192, 64, kernel_size=5, padding=2)),    # 8
            ('relu1', nn.ReLU(inplace=True)),                           # 9
            ('unpool1', nn.MaxUnpool2d(kernel_size=3, stride=2)),       # 10
            ('deconv0', nn.ConvTranspose2d(unet_multp * 64, 1, kernel_size=11, stride=4, padding=2, output_padding=1)),
            ('tanh', nn.Tanh())
        ]

        if batch_norm:
            base_archi.insert(9, ('norm1', nn.BatchNorm2d(64)))
            base_archi.insert(6, ('norm2', nn.BatchNorm2d(192)))
            base_archi.insert(4, ('norm3', nn.BatchNorm2d(384)))
            base_archi.insert(2, ('norm4', nn.BatchNorm2d(256)))

        if end_relu:
            base_archi.append(('relu0', nn.ReLU(inplace=True)))

        self.base_archi = coll.OrderedDict(base_archi)

        self.deconv = nn.Sequential(self.base_archi)
        logger.info('Final feature extractor architecture:')
        logger.info(self.deconv)
        self.down_ratio = {
            'unpool4': 6/13,
            'conv4': 6/13,
            'relu4': 6/13,
            'conv3': 6/13,
            'relu3': 6/13,
            'conv2': 6/13,
            'relu2': 6/13,
            'unpool2': 6/27,
            'conv1': 6/27,
            'relu1': 6/27,
            'unpool1': 6/55,
            'deconv0': 6/224,
        }

    def forward(self, x, **kwargs):
        ind = kwargs.pop('id', None)
        res = kwargs.pop('res', None)

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        maps = dict()

        for name, lay in self.deconv.named_children():
            if name == 'unpool4':
                x = lay(x, ind[2])
            elif name == 'unpool2':
                x = lay(x, ind[1])
            elif name == 'unpool1':
                x = lay(x, ind[0])
            elif self.res and name == 'relu2':
                x = lay(x + res[1])
            elif self.res and name == 'relu1':
                x = lay(x + res[0])
            elif self.unet and name == 'conv1':
                x = lay(torch.cat([x, res[1]], dim=1))
            elif self.unet and name == 'deconv0':
                x = lay(torch.cat([x, res[0]], dim=1))
            else:
                x = lay(x)

            maps[name] = x

        maps['maps'] = x

        return maps

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


if __name__ == '__main__':
    tensor_input = torch.rand([10, 3, 224, 224]).cuda()
    net = Feat().cuda()
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

    conv = Feat(indices=True, res=True).cuda()
    deconv = Deconv(res=True).cuda()

    output_conv = conv(auto.Variable(tensor_input).cuda())
    returned_maps = deconv(output_conv['output'],
                  id1=output_conv['id'][0],
                  id2=output_conv['id'][1],
                  id3=output_conv['id'][2],
                  res1=output_conv['res'][0],
                  res2=output_conv['res'][1]
                  )

    print(returned_maps['deconv0'].size())
    print(returned_maps.keys())
