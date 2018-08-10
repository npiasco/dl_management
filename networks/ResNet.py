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


class Feat(models.ResNet):
    def __init__(self, **kwargs):
        num_layer = kwargs.pop('num_layer', 18)
        self.end_relu = kwargs.pop('end_relu', False)
        load_imagenet = kwargs.pop('load_imagenet', True)
        self.layers_to_train = kwargs.pop('layers_to_train', 'up_to_lay3')
        self.indices = kwargs.pop('indices', False)
        self.res = kwargs.pop('res', False)
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
            del self.layer4

        logger.info('Final feature extractor architecture:')
        logger.info(self)


    def forward(self, x):

        if hasattr(self, 'jet_tf'):
            x = self.jet_tf(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

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

        """
        if self.unet or self.res:
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
        """
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
                                [self.layer4,]]
        elif layers_to_train == 'up_to_conv3':
            training_params = [{'params': layers.parameters()} for layers in
                               [self.layer4, self.layer3]]
        elif layers_to_train == 'up_to_conv2':
            training_params = [{'params': layers.parameters()} for layers in
                               [self.layer4, self.layer3, self.layer2]]
        elif layers_to_train == 'up_to_conv1':
            training_params = [{'params': layers.parameters()} for layers in
                               [self.layer4, self.layer3, self.layer2, self.layer1]]
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



if __name__ == '__main__':
    tensor_input = torch.rand([10, 3, 224, 224]).cuda()
    net = Feat(num_layer=18, truncated=2).cuda()
    feat_output = net(auto.Variable(tensor_input))
    print(feat_output)
