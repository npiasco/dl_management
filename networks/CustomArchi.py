import setlog
import torch.nn as nn
import torch
import networks.Descriptor as Desc
import networks.ResNet as ResNet
import networks.FeatAggregation as Agg
import copy
import collections as coll


logger = setlog.get_logger(__name__)


class DeploymentNet(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.main_desc = Desc.Main(
            agg_method='NetVLAD',
            agg_method_param={
                'bias': True,
                'cluster_size': 64,
                'feature_size': 256
            },
            base_archi='Resnet',
            base_archi_param={
                'end_relu': False,
                'num_layer': 18,
                'truncated': 3,
                'unet': True,
                'load_imagenet': False
            }
        )

        self.aux_desc = Desc.Main(
            agg_method='NetVLAD',
            agg_method_param={
                'bias': True,
                'cluster_size': 64,
                'feature_size': 256
            },
            base_archi='Alexnet',
            base_archi_param={
                'end_relu': False,
                'jet_tf': True,
                'load_imagenet': False
            }
        )

        self.deconv = ResNet.Deconv(
            size_res_1=128,
            size_res_2=64,
            input_size=256,
            up_factor=1
        )

        self.fuze = Agg.Concat(aux_ratio=1.0, main_ratio=1.0, norm=False)

    def forward(self, x):
        self.eval()

        main_output = self.main_desc(x)

        map = self.deconv(main_output['feat'], main_output['res_1'], main_output['res_2'])

        aux_desc = self.aux_desc(map)

        desc = self.fuze(main_output['desc'], aux_desc)

        return desc

class PixEncoder(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        self.layers_to_train = kwargs.pop('layers_to_train', 'all')
        i_channel = kwargs.pop('input_channels', 3)
        d_fact = kwargs.pop('d_fact', 1)
        k_size = kwargs.pop('k_size', 2)
        norm_layer = kwargs.pop('norm_layer', 'group')

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if norm_layer == 'group':
            norm_layer_func = lambda x: copy.deepcopy(nn.GroupNorm(x//2, x))
        elif norm_layer == 'batch':
            norm_layer_func = lambda x: copy.deepcopy(nn.BatchNorm2d(x))


        base_archi = [
            ('conv0', nn.Conv2d(i_channel, int(64 / d_fact), kernel_size=k_size*4, stride=2, padding=(k_size*4-1)//2)),
            ('relu0', nn.LeakyReLU(0.2, inplace=True)),
            ('conv1', nn.Conv2d(int(64 / d_fact), int(128 / d_fact), kernel_size=k_size+1, stride=1, padding=(k_size+1)//2)),
            ('bn1', norm_layer_func(int(128/d_fact))),
            ('relu1', nn.LeakyReLU(0.2, inplace=True)),
            ('conv2', nn.Conv2d(int(128 / d_fact), int(256 / d_fact), kernel_size=k_size, stride=2, padding=(k_size-1)//2)),
            ('bn2', norm_layer_func(int(256 / d_fact))),
            ('relu2', nn.LeakyReLU(0.2, inplace=True)),
            ('conv3', nn.Conv2d(int(256 / d_fact), int(512 / d_fact), kernel_size=k_size+1, stride=1, padding=(k_size+1)//2)),
            ('bn3', norm_layer_func(int(512 / d_fact))),
            ('relu3', nn.LeakyReLU(0.2, inplace=True)),
            ('conv4', nn.Conv2d(int(512 / d_fact), int(512 / d_fact), kernel_size=k_size, stride=2, padding=(k_size-1)//2)),
            ('bn4', norm_layer_func(int(512 / d_fact))),
            ('relu4', nn.LeakyReLU(0.2, inplace=True)),
            ('conv5', nn.Conv2d(int(512 / d_fact), int(512 / d_fact), kernel_size=k_size+1, stride=1, padding=(k_size+1)//2)),
            ('bn5', norm_layer_func(int(512 / d_fact))),
            ('relu5', nn.LeakyReLU(0.2, inplace=True)),
            ('conv6', nn.Conv2d(int(512 / d_fact), int(512 / d_fact), kernel_size=k_size, stride=2, padding=(k_size-1)//2)),
            ('bn6', norm_layer_func(int(512 / d_fact))),
            ('relu6', nn.LeakyReLU(0.2, inplace=True)),
            ('conv7', nn.Conv2d(int(512 / d_fact), int(512 / d_fact), kernel_size=k_size+1, stride=1, padding=(k_size+1)//2)),
            ('bn7', norm_layer_func(int(512 / d_fact))),
            ('relu7', nn.LeakyReLU(0.2, inplace=True)),
        ]

        self.feature = nn.Sequential(
            coll.OrderedDict(base_archi)
        )

        logger.info('Final architecture is:')
        logger.info(self.feature)

    def forward(self, x):
        output = dict()
        for name, lay in self.feature.named_children():
            x = lay(x)
            output[name] = x
        return output

    def get_training_layers(self, layers_to_train=None):
        if layers_to_train is None:
            layers_to_train = self.layers_to_train
        if layers_to_train == 'all':
            return [{'params': self.feature.parameters()}]
        elif layers_to_train == 'no':
            return []

    def full_save(self, discard_tf=False):
        if discard_tf:
            raise NotImplementedError('Functionality not implemented')
        return {'feature': self.feature.state_dict(),}


class PixDecoder(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        self.layers_to_train = kwargs.pop('layers_to_train', 'all')
        out_channel = kwargs.pop('out_channel', 1)
        d_fact = kwargs.pop('d_fact', 1)
        k_size = kwargs.pop('k_size', 2)
        norm_layer = kwargs.pop('norm_layer', 'group')
        div_fact = kwargs.pop('div_fact', 1)

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if div_fact not in [1, 2, 4]:
            raise ValueError('Output is not divisible by {}'.format(div_fact))

        if norm_layer == 'group':
            norm_layer_func = lambda x: copy.deepcopy(nn.GroupNorm(x//2, x))
        elif norm_layer == 'batch':
            norm_layer_func = lambda x: copy.deepcopy(nn.BatchNorm2d(x))

        #TODO: construct various archi depending on the desired scale factor according to the output size

        base_archi = [
            ('conv7', nn.Conv2d(int(512 / d_fact), int(512 / d_fact), kernel_size=k_size + 1, stride=1,
                                         padding=(k_size + 1) // 2)),
            ('bn7', norm_layer_func(int(512 / d_fact))),
            ('relu7', nn.ReLU(inplace=True)),
            ('conv6', nn.ConvTranspose2d(int(512 / d_fact * 2), int(512 / d_fact), kernel_size=k_size, stride=2, padding=(k_size-1)//2)),
            ('bn6', norm_layer_func(int(512 / d_fact))),
            ('relu6', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(int(512 / d_fact * 2), int(512 / d_fact), kernel_size=k_size + 1, stride=1,
                                         padding=(k_size + 1) // 2)),
            ('bn5', norm_layer_func(int(512 / d_fact))),
            ('relu5', nn.ReLU(inplace=True)),
            ('conv4', nn.ConvTranspose2d(int(512 / d_fact * 2), int(512 / d_fact), kernel_size=k_size, stride=2, padding=(k_size-1)//2)),
            ('bn4', norm_layer_func(int(512 / d_fact))),
            ('relu4', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(int(512 / d_fact * 2), int(256 / d_fact), kernel_size=k_size + 1, stride=1,
                                         padding=(k_size + 1) // 2)),
            ('bn3', norm_layer_func(int(256 / d_fact))),
            ('relu3', nn.ReLU(inplace=True)),
        ]

        end_div_4 = [
            ('conv2', nn.Conv2d(int(256 / d_fact * 2), int(out_channel), kernel_size=k_size*2+1, stride=1,
                                padding=(k_size*2+1) // 2)),
        ]

        layer_21 = [
            ('conv2', nn.ConvTranspose2d(int(256 / d_fact * 2), int(128 / d_fact), kernel_size=k_size, stride=2,
                                         padding=(k_size - 1) // 2)),
            ('bn2', norm_layer_func(int(128 / d_fact))),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(int(128 / d_fact * 2), int(64 / d_fact), kernel_size=k_size + 1, stride=1,
                                padding=(k_size + 1) // 2)),
            ('bn1', norm_layer_func(int(64 / d_fact))),
            ('relu1', nn.ReLU(inplace=True)),
        ]

        end_div_2 = layer_21 + [
            ('conv0', nn.Conv2d(int(64 / d_fact * 2), int(out_channel), kernel_size=(k_size*2+1), stride=1,
                                padding=(k_size*2+1) // 2)),
        ]

        end_div_1 = layer_21 + [
            ('conv0', nn.ConvTranspose2d(int(64 / d_fact * 2), int(out_channel), kernel_size=k_size*2, stride=2,
                                padding=(k_size*2 - 1) // 2)),
        ]

        end = [
            ('sig', nn.Sigmoid()),
        ]

        if div_fact == 1:
            base_archi = base_archi + end_div_1 + end
        elif div_fact == 2:
            base_archi = base_archi + end_div_2 + end
        elif div_fact == 4:
            base_archi = base_archi + end_div_4 + end

        self.feature = nn.Sequential(
            coll.OrderedDict(base_archi)
        )

        logger.info('Final architecture is:')
        logger.info(self.feature)

    def forward(self, unet):
        for name, lay in self.feature.named_children():
            if name == 'conv7':
                x = lay(unet[name])
            else:
                if 'conv' in name:
                    x = torch.cat((x, unet[name]), dim=1)
                x = lay(x)
        return x

    def get_training_layers(self, layers_to_train=None):
        if layers_to_train is None:
            layers_to_train = self.layers_to_train
        if layers_to_train == 'all':
            return [{'params': self.feature.parameters()}]
        elif layers_to_train == 'no':
            return []

    def full_save(self, discard_tf=False):
        if discard_tf:
            raise NotImplementedError('Functionality not implemented')
        return {'feature': self.feature.state_dict(), }


class Softlier(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        input_size = kwargs.pop('input_size', 256)
        size_layer = kwargs.pop('size_layer', 256)
        num_inter_layers = kwargs.pop('num_inter_layers', 1)
        self.consensus = kwargs.pop('consensus', False)
        self.layers_to_train = kwargs.pop('layers_to_train', 'all')

        if kwargs:
            logger.error('Unexpected **kwargs: %r' % kwargs)
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        base_archi = list()

        if num_inter_layers:
            base_archi += [
                ('fc0', nn.Linear(input_size, size_layer)),
                ('relu0',  nn.ReLU(inplace=True)),
            ]
            for layers in self.intermediate_layers(num_inter_layers - 1, size_layer):
                base_archi.append(layers)
            base_archi+= [
                ('thresh', nn.Linear(size_layer, 1)),
            ]
        else:
            base_archi += [
                ('thresh', nn.Linear(size_layer, 1)),
            ]

        self.base_archi = coll.OrderedDict(base_archi)
        self.thresholder = nn.Sequential(self.base_archi)
        logger.info('Final thresholder architecture:')
        logger.info(self.thresholder)

    @staticmethod
    def intermediate_layers(num_layer, size_layer):
        lays = (
            ('fc', nn.Linear(size_layer, size_layer)),
            ('relu', nn.ReLU(inplace=True)),
        )
        for i in range(1, num_layer+1):
            for lay in lays:
                yield (lay[0] + str(i), copy.deepcopy(lay[1]))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        if self.consensus:
            x = torch.sigmoid(self.thresholder(x))
            mean_dist = torch.mean(x, 0)
            eps = 1e-5
            x = torch.sigmoid((x - mean_dist + eps)*1e10)
        else:
            x = torch.sigmoid((self.thresholder(x))*1e10)
        return x

    def get_training_layers(self, layers_to_train=None):
        def sub_layers(name):
            return {
                'all': [{'params': self.thresholder.parameters()}]
            }.get(name)

        if not layers_to_train:
            layers_to_train = self.layers_to_train
        return sub_layers(layers_to_train)


if __name__ == '__main__':
    input_size = 224//2
    tensor_input = torch.rand([1, 3, input_size, input_size])
    '''
    net = DeploymentNet()

    root = '/mnt/anakim/data/RGBtrainD/Resnet18T/BUTF/OTS/2NetVLAD/'
    net.main_desc.feature.load_state_dict(torch.load(root + 'Main_feature.pth'))
    net.main_desc.descriptor.load_state_dict(torch.load(root + 'Main_descriptor.pth'))
    net.deconv.deconv_1.load_state_dict(torch.load(root + 'Deconv_deconv1.pth'))
    net.deconv.deconv_2.load_state_dict(torch.load(root + 'Deconv_deconv2.pth'))
    net.deconv.deconv_3.load_state_dict(torch.load(root + 'Deconv_deconv3.pth'))
    net.aux_desc.feature.load_state_dict(torch.load(root + 'Aux_feature.pth'))
    net.aux_desc.descriptor.load_state_dict(torch.load(root + 'Aux_descriptor.pth'))

    torch.save(net.state_dict(), 'default.pth')
    '''
    enc = PixEncoder(k_size=4, d_fact=2)
    dec= PixDecoder(k_size=4, d_fact=2, out_channel=1, div_fact=2)

    feat_output = enc(tensor_input)
    output = dec(feat_output)
    print(output.size())
    print(feat_output['conv1'].size())
    #print([res.size() for res in feat_output.values()])

    desc_feat = torch.rand(50, 64)
    softlier = Softlier(input_size=64, num_inter_layers=1)
    print(softlier(desc_feat))
