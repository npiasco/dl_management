import setlog
import torch.nn as nn
import collections as coll
import torch.autograd as auto
import torch
import torchvision.models as models
import os
import networks.CustomLayers as custom
import copy


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
        i_channel = kwargs.pop('input_channels', 3)
        jet_tf = kwargs.pop('jet_tf', False)
        jet_tf_param = kwargs.pop('jet_tf_param', dict())
        mean_pooling = kwargs.pop('mean_pooling', False)
        leaky_relu = kwargs.pop('leaky_relu', False)
        norm_layer = kwargs.pop('norm_layer', 'batch')

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        polling_layer_param = {
            'kernel_size': 3,
            'stride': 2,
        }

        polling_layer = nn.AvgPool2d(**polling_layer_param) if mean_pooling else \
            nn.MaxPool2d(**polling_layer_param, return_indices=self.indices)
        relu_type = nn.LeakyReLU(inplace=True, negative_slope=0.02) if leaky_relu else nn.ReLU(inplace=True)
        if norm_layer == 'group':
            norm_layer_func = lambda x: copy.deepcopy(nn.GroupNorm(x // 2, x))
        elif norm_layer == 'batch':
            norm_layer_func = lambda x: copy.deepcopy(nn.BatchNorm2d(x))

        base_archi = [
            ('conv0', nn.Conv2d(i_channel, 64, kernel_size=11, stride=4, padding=2)),   # 0
            ('relu0', copy.deepcopy(relu_type)),                                   # 1
            ('pool0', copy.deepcopy(polling_layer)),                   # 2
            ('conv1', nn.Conv2d(64, 192, kernel_size=5, padding=2)),            # 3
            ('relu1', copy.deepcopy(relu_type)),                                   # 4
            ('pool1', copy.deepcopy(polling_layer)),                   # 5
            ('conv2', nn.Conv2d(192, 384, kernel_size=3, padding=1)),           # 6
            ('relu2', copy.deepcopy(relu_type)),                                   # 7
            ('conv3', nn.Conv2d(384, 256, kernel_size=3, padding=1)),           # 8
            ('relu3', copy.deepcopy(relu_type)),                                   # 9
            ('conv4', nn.Conv2d(256, 256, kernel_size=3, padding=1)),           # 10
            ]

        if batch_norm:
            base_archi.insert(9, ('norm3', norm_layer_func(256)))
            base_archi.insert(7, ('norm2', norm_layer_func(384)))
            base_archi.insert(4, ('norm1', norm_layer_func(192)))
            base_archi.insert(1, ('norm0', norm_layer_func(64)))

        if end_relu:
            base_archi.append(('relu4', copy.deepcopy(relu_type)))

        if end_max_polling:
            base_archi.append(('pool3', copy.deepcopy(polling_layer)))

        if jet_tf:
            base_archi = [('jet_tf', custom.IndexEmbedding(**jet_tf_param))] + base_archi

        self.base_archi = coll.OrderedDict(base_archi)
        self.feature = nn.Sequential(self.base_archi)
        logger.info('Final feature extractor architecture:')
        logger.info(self.feature)
        self._down_ratio = 0

        if load_imagenet:
            self.load(mono)

    def forward(self, x):
        if self.indices or self.res or self.unet:
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
            if self.unet:
                return {'output': x, 'feat': feat, 'res_1': res[0], 'res_2': res[1], 'id': ind, 'res': res}
            else:
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
        if layers_to_train is None:
            layers_to_train = self.layers_to_train
        if layers_to_train == 'all':
            training_params = [{'params': layers.parameters()} for layers in list(self.feature.children())]
        elif layers_to_train == 'up_to_conv4':
            training_params = [{'params': layers.parameters()} for layers in
                                list(self.feature.children())[list(self.base_archi.keys()).index('conv4'):]]
        elif layers_to_train == 'up_to_conv3':
            training_params = [{'params': layers.parameters()} for layers in
                               list(self.feature.children())[list(self.base_archi.keys()).index('conv3'):]]
        elif layers_to_train == 'up_to_conv2':
            training_params = [{'params': layers.parameters()} for layers in
                                list(self.feature.children())[list(self.base_archi.keys()).index('conv2'):]]
        elif layers_to_train == 'up_to_conv1':
            training_params = [{'params': layers.parameters()} for layers in
                                list(self.feature.children())[list(self.base_archi.keys()).index('conv1'):]]
        elif layers_to_train == 'only_jet':
            training_params = [{'params':
                                    list(self.feature.children())[
                                        list(self.base_archi.keys()).index('jet_tf')
                                    ].parameters()}]
        elif layers_to_train == 'no_layer':
            training_params = []
        else:
            raise KeyError('No behaviour for layers_to_train = {}'.format(layers_to_train))

        return training_params

    @property
    def down_ratio(self):
        if 'pool3' in [elem[0] for elem in list(self.feature.named_children())]:
            self._down_ratio = 224 / 6
        else:
            self._down_ratio = 224 / 13
        return self._down_ratio

    def final_feat_num(self):
        return 256


class Deconv(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        batch_norm = kwargs.pop('batch_norm', False)
        end_relu = kwargs.pop('end_relu', False)
        modality_ch = kwargs.pop('modality_ch', 1)
        upsample = kwargs.pop('upsample', False)
        self.layers_to_train = kwargs.pop('layers_to_train', 'all')
        self.res = kwargs.pop('res', False)
        self.unet = kwargs.pop('unet', False)
        smooth = kwargs.pop('smooth', False)
        final_jet_tf = kwargs.pop('final_jet_tf', False)
        jet_tf_is_trainable = kwargs.pop('jet_tf_is_trainable', False)
        leaky_relu = kwargs.pop('leaky_relu', False)

        relu_type = nn.LeakyReLU(inplace=True, negative_slope=0.02) if leaky_relu else nn.ReLU(inplace=True)

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        unet_multp = 2 if self.unet else 1

        base_archi = [
            ('unpool4', nn.MaxUnpool2d(kernel_size=3, stride=2)),       # 0
            ('conv4', nn.Conv2d(256, 256, kernel_size=3, padding=1)),   # 1
            ('relu4', copy.deepcopy(relu_type)),                           # 2
            ('conv3', nn.Conv2d(256, 384, kernel_size=3, padding=1)),   # 3
            ('relu3', copy.deepcopy(relu_type)),                           # 4
            ('conv2', nn.Conv2d(384, 192, kernel_size=3, padding=1)),   # 5
            ('relu2', copy.deepcopy(relu_type)),                           # 6
            ('unpool2', nn.MaxUnpool2d(kernel_size=3, stride=2)),       # 7
            ('conv1', nn.Conv2d(unet_multp * 192, 64, kernel_size=5, padding=2)),    # 8
            ('relu1', copy.deepcopy(relu_type)),                           # 9
            ('unpool1', nn.MaxUnpool2d(kernel_size=3, stride=2)),       # 10
        ]

        if upsample:
            base_archi += [
                ('deconv0', nn.ConvTranspose2d(unet_multp * 64, modality_ch, kernel_size=6, stride=2, padding=2,
                                               output_padding=1)),
                ('deconv1', nn.ConvTranspose2d(modality_ch, modality_ch, kernel_size=7, stride=2, padding=2,
                                               output_padding=1)),
                ('tanh', nn.Tanh())
            ]
        else:
            base_archi += [
                ('deconv0', nn.ConvTranspose2d(unet_multp * 64, modality_ch, kernel_size=11, stride=4, padding=2,
                                               output_padding=1)),
                ('tanh', nn.Tanh())
            ]

        if batch_norm:
            base_archi.insert(9, ('norm1', nn.BatchNorm2d(64)))
            base_archi.insert(6, ('norm2', nn.BatchNorm2d(192)))
            base_archi.insert(4, ('norm3', nn.BatchNorm2d(384)))
            base_archi.insert(2, ('norm4', nn.BatchNorm2d(256)))

        if end_relu:
            base_archi.append(('relu0', copy.deepcopy(relu_type)))

        if final_jet_tf:
            base_archi.append(('jet_tf', custom.IndexEmbedding(num_embedding=256,
                                                               size_embedding=3,
                                                               init_jet=True,
                                                               trainable=jet_tf_is_trainable)))

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
            elif self.unet and name in ('deconv0', 'upsample1'):
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
    tensor_input = torch.rand([10, 3, 56, 56])
    #net = Feat(unet=True, indices=True, norm_layer='group')
    net = Feat(norm_layer='group')
    feat_output = net(auto.Variable(tensor_input))
    print(feat_output['feat'].size(),feat_output['res_1'].size(),feat_output['res_2'].size())
    import networks.ResNet as RNet
    deconv = RNet.Deconv(size_res_1=192, alexnet_entry=True, reduce_factor=4, norm_layer='group', final_activation='sig', extended_size=True)
    map = deconv(feat_output['feat'], feat_output['res_1'], feat_output['res_2'])
    print(map.size())
    '''
    net = Feat(batch_norm=False, end_relu=True).cuda()
    feat_output = net(auto.Variable(tensor_input).cuda())
    net.eval()
    feat_output = net(auto.Variable(tensor_input).cuda())
    print(feat_output[0])
    net.layers_to_train = 'up_to_conv2'
    print(net.get_training_layers())
    print(net.down_ratio)
    net = Feat(batch_norm=False, end_relu=True, end_max_polling=True, mean_pooling=True, leaky_relu=True).cuda()
    feat_output = net(auto.Variable(tensor_input).cuda())
    print(feat_output.size())
    print(net.down_ratio)

    conv = Feat(indices=True, res=True, end_max_polling=True).cuda()
    deconv = Deconv(res=True, smooth=True).cuda()

    output_conv = conv(auto.Variable(tensor_input).cuda())

    returned_maps = deconv(output_conv['output'],
                  id=output_conv['id'],
                  res=output_conv['res'],
                  )

    print(returned_maps['maps'].size())
    #print(returned_maps.keys())
    '''