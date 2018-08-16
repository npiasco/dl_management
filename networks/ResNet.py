import setlog
import torch.autograd as auto
import torch
import torchvision.models as models
import os
import networks.CustomLayers as custom
import types
import torch.nn as nn
import torch.nn.functional as func
import collections as coll


logger = setlog.get_logger(__name__)


class Feat(models.ResNet):
    def __init__(self, **kwargs):
        num_layer = kwargs.pop('num_layer', 18)
        self.end_relu = kwargs.pop('end_relu', False)
        load_imagenet = kwargs.pop('load_imagenet', True)
        self.layers_to_train = kwargs.pop('layers_to_train', 'up_to_conv3')
        self.unet = kwargs.pop('unet', False)
        self.truncated = kwargs.pop('truncated', False)
        jet_tf = kwargs.pop('jet_tf', False)
        jet_tf_is_trainable = kwargs.pop('jet_tf_is_trainable', False)

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if num_layer == 18:
            models.ResNet.__init__(self, models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=1)
            self.n_block_last_layer = 2
            self._final_feat_num = 512
        elif num_layer == 34:
            models.ResNet.__init__(self, models.resnet.BasicBlock, [3, 4, 6, 3], num_classes=1)
            self.n_block_last_layer = 2
            self._final_feat_num = 512
        elif num_layer == 50:
            models.ResNet.__init__(self, models.resnet.Bottleneck, [3, 4, 6, 3], num_classes=1)
            self.n_block_last_layer = 3
            self._final_feat_num = 2048
        else:
            raise ValueError('No resnet with {} layers.'.format(num_layer))

        del self.fc
        del self.avgpool

        if load_imagenet:
            self.load(num_layer)

        if jet_tf:
            self.jet_tf = custom.IndexEmbedding(num_embedding=256,
                                                size_embedding=3,
                                                trainable=jet_tf_is_trainable)

        if self.truncated:
            self.endlayer = getattr(self, 'layer{}'.format(self.truncated))
            for i in range(self.truncated,4):
                delattr(self, 'layer{}'.format(i))
                setattr(self, 'layer{}'.format(i), lambda x: x)
                self._final_feat_num /= 2
        else:
            self.endlayer = self.layer4
        del self.layer4

        logger.info('Final feature extractor architecture:')
        logger.info(self)


    def forward(self, x):

        if hasattr(self, 'jet_tf'):
            x = self.jet_tf(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x_res_1 = self.maxpool(x)

        x = self.layer1(x_res_1)
        x = self.layer2(x)
        x_res_2 = self.layer3(x)
        x = x_res_2

        # Lay 4
        for i in range(self.n_block_last_layer-1):
            x = self.endlayer[i](x)

        i = self.n_block_last_layer-1

        residual = x

        x = self.endlayer[i].conv1(x)
        x = self.endlayer[i].bn1(x)
        x = self.endlayer[i].relu(x)

        x = self.endlayer[i].conv2(x)
        x = self.endlayer[i].bn2(x)

        if self.n_block_last_layer == 3:
            x = self.endlayer[i].relu(x)

            x = self.endlayer[i].conv3(x)
            x = self.endlayer[i].bn3(x)

        x += residual
        if self.end_relu:
            x = self.relu(x)


        if self.unet:
            return {'feat': x, 'res_1': x_res_1, 'res_2': x_res_2}
        else:
            return x

    def load(self, num_layer):
        name = 'resnet{}_ots.pth'.format(num_layer)
        logger.info('Loading pretrained weight ' + os.environ['CNN_WEIGHTS'] + name)
        self.load_state_dict(torch.load(os.environ['CNN_WEIGHTS'] + name))


    def get_training_layers(self, layers_to_train=None):
        if layers_to_train is None:
            layers_to_train = self.layers_to_train
        if layers_to_train == 'all':
            training_params = [{'params': self.parameters()} ]
        elif layers_to_train == 'up_to_conv4':
            training_params = [{'params': layers.parameters()} for layers in
                                [self.endlayer,]]
        elif layers_to_train == 'up_to_conv3':
            training_params = [{'params': layers.parameters()} for layers in
                               [self.endlayer, self.layer3] if not isinstance(layers, types.LambdaType)]
        elif layers_to_train == 'up_to_conv2':
            training_params = [{'params': layers.parameters()} for layers in
                               [self.endlayer, self.layer3, self.layer2] if not isinstance(layers, types.LambdaType)]
        elif layers_to_train == 'up_to_conv1':
            training_params = [{'params': layers.parameters()} for layers in
                               [self.endlayer, self.layer3, self.layer2, self.layer1] if not isinstance(layers, types.LambdaType)]
        elif layers_to_train == 'only_jet':
            training_params = [{'params': self.jet_tf.parameters()}]
        else:
            raise KeyError('No behaviour for layers_to_train = {}'.format(layers_to_train))

        return training_params

    @property
    def down_ratio(self):
        return 224/7

    def final_feat_num(self):
        return self._final_feat_num

class Deconv(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.layers_to_train = kwargs.pop('layers_to_train', 'all')
        modality_ch = kwargs.pop('modality_ch', 1)
        size_res_1 = kwargs.pop('size_res_1', 128)
        size_res_2 = kwargs.pop('size_res_2', 64)
        input_size = kwargs.pop('input_size', 256)
        self.up_factor = kwargs.pop('up_factor', 1)
        alexnet_entry = kwargs.pop('alexnet_entry', False)

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        out_pad = 1 if alexnet_entry else 0

        base_archi = {
            1:
                [
                    ('deconv4', nn.ConvTranspose2d(input_size, 256, kernel_size=4, stride=2, padding=1,
                                                   output_padding=out_pad)),
                    ('norm4', nn.BatchNorm2d(256)),
                    ('relu4', nn.LeakyReLU(inplace=True, negative_slope=0.02)),  # 2
                    ('conv3', nn.Conv2d(256, 384, kernel_size=3, padding=1)),  # 3
                    ('norm3', nn.BatchNorm2d(384)),
                    ('relu3', nn.LeakyReLU(inplace=True, negative_slope=0.02)),  # 4
                    ('conv2', nn.Conv2d(384, size_res_1, kernel_size=3, padding=1)),  # 5
                    ('norm2', nn.BatchNorm2d(size_res_1)),
                    ('relu2', nn.LeakyReLU(inplace=True, negative_slope=0.02)),  # 6
                ],
            2:
                [
                    ('deconv1', nn.ConvTranspose2d(2 * size_res_1, 64, kernel_size=4, stride=2, padding=1,
                                                   output_padding=out_pad)),
                    ('norm1', nn.BatchNorm2d(64)),
                    ('relu1', nn.LeakyReLU(inplace=True, negative_slope=0.02)),  # 9
                ],
            3:
                [
                    ('deconv0', nn.ConvTranspose2d(2 * size_res_2, modality_ch, kernel_size=6, stride=2, padding=2-out_pad)),
                    ('deconvf', nn.ConvTranspose2d(modality_ch, modality_ch, kernel_size=6, stride=2, padding=2,
                                                   output_padding=0)),
                    ('tanh', nn.Tanh()),

                ]
        }
        self.base_archi = dict()
        self.base_archi[1] = coll.OrderedDict(base_archi[1])
        self.base_archi[2] = coll.OrderedDict(base_archi[2])
        self.base_archi[3] = coll.OrderedDict(base_archi[3])

        self.deconv_1 = nn.Sequential(self.base_archi[1])
        self.deconv_2 = nn.Sequential(self.base_archi[2])
        self.deconv_3 = nn.Sequential(self.base_archi[3])
        logger.info('Final deconv module architecture:')
        logger.info(self)
        self.down_ratio = {
            'unpool4': 6 / 13,
            'conv4': 6 / 13,
            'relu4': 6 / 13,
            'conv3': 6 / 13,
            'relu3': 6 / 13,
            'conv2': 6 / 13,
            'relu2': 6 / 13,
            'unpool2': 6 / 27,
            'conv1': 6 / 27,
            'relu1': 6 / 27,
            'unpool1': 6 / 55,
            'deconv0': 6 / 224,
        }

    def forward(self, x, res1, res2):
        if self.up_factor > 1:
            x = func.upsample(x, scale_factor=self.up_factor)
            res2 = func.upsample(res2, scale_factor=self.up_factor)
        x = self.deconv_1(x)
        print(x.size(), res2.size())
        x = torch.cat((x, res2), dim=1)
        x = self.deconv_2(x)
        print(x.size(), res1.size())
        x = torch.cat((x, res1), dim=1)
        map = self.deconv_3(x)

        return map


    def get_training_layers(self, layers_to_train=None):
        if layers_to_train is None:
            layers_to_train = self.layers_to_train
        if layers_to_train == 'all':
            training_params = [{'params': self.parameters()}]
        else:
            raise KeyError('No behaviour for layers_to_train = {}'.format(layers_to_train))

        return training_params


if __name__ == '__main__':
    tensor_input = torch.rand([10, 3, 224, 224])
    net = Feat(num_layer=18, truncated=3, load_imagenet=True, unet=True)
    feat_output = net(auto.Variable(tensor_input))
    print(feat_output['feat'].size(),feat_output['res_1'].size(),feat_output['res_2'].size())

    #deconvnet = Deconv(size_res_1=256, input_size=512, up_factor=2)
    deconvnet = Deconv(size_res_1=128)
    map = deconvnet(feat_output['feat'], feat_output['res_1'], feat_output['res_2'])
