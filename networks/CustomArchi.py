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
                'jet_tf_param': {'amplitude': 2.0, 'min_value': -1.0},
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


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        nn.Module.__init__(self)
        self.scale_factor = kwargs.pop('scale_factor', 2)
        kernel_size = kwargs.pop('kernel_size', 3)
        padding = kwargs.pop('padding', 0)
        stride = kwargs.pop('stride', 1)
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride)

    def forward(self, x):
        upx = nn.functional.interpolate(x, scale_factor=self.scale_factor)
        x = self.conv(upx)

        return x


class PixelRnn(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        nn.Module.__init__(self)
        rec_module = kwargs.pop('rec_module', 'gru')
        num_layers = kwargs.pop('num_layers', 1)
        bidirectional = kwargs.pop('bidirectional', True)
        self.padding = kwargs.pop('padding', 0)
        self.kernel_size = kwargs.pop('kernel_size', 1)
        self.stride = kwargs.pop('stride', 1)
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if rec_module == 'gru':
            self.h_rec_module = nn.GRU(input_size=in_channels, hidden_size=out_channels,
                                       num_layers=num_layers, bidirectional=bidirectional)
            self.v_rec_module = nn.GRU(input_size=in_channels, hidden_size=out_channels,
                                       num_layers=num_layers, bidirectional=bidirectional)
        elif rec_module == 'lstm':
            self.h_rec_module = nn.LSTM(input_size=in_channels, hidden_size=out_channels,
                                        num_layers=num_layers, bidirectional=bidirectional)
            self.v_rec_module = nn.LSTM(input_size=in_channels, hidden_size=out_channels,
                                        num_layers=num_layers, bidirectional=bidirectional)
        else:
            raise NotImplementedError('No recurent module named {}'.format(rec_module))

        self.h0_size = (num_layers*(2 if bidirectional else 1), out_channels)

    def forward(self, x):
        output_h = list()
        output_v = list()
        for line in x.transpose(0, 2).transpose(1, 3):
            output, _ = self.h_rec_module(line)
            output_h.append(output.transpose(0, 1).transpose(1, 2))
        for col in x.transpose(0, 2).transpose(1, 3).transpose(0, 1):
            output, _ = self.v_rec_module(col)
            output_v.append(output.transpose(0, 1).transpose(1, 2))

        x = torch.cat((torch.stack(output_h, dim=2), torch.stack(output_v, dim=3)), dim=1)
        return x


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
        dropout = kwargs.pop('dropout', False)

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

        if dropout:
            base_archi.append(('dp2', nn.Dropout2d(dropout)))
            base_archi.insert(12, ('dp3', nn.Dropout2d(dropout)))
            base_archi.insert(9, ('dp4', nn.Dropout2d(dropout)))
            base_archi.insert(6, ('dp5', nn.Dropout2d(dropout)))
            base_archi.insert(3, ('dp6', nn.Dropout2d(dropout)))

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


class PixDecoderMultiscale(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        self.layers_to_train = kwargs.pop('layers_to_train', 'all')
        d_fact = kwargs.pop('d_fact', 1)
        k_size = kwargs.pop('k_size', 2)
        norm_layer = kwargs.pop('norm_layer', 'group')
        div_fact = kwargs.pop('div_fact', 1)
        pixel_rnn = kwargs.pop('pixel_rnn', False)
        rnn_type = kwargs.pop('rnn_type', 'gru')

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if div_fact not in [1, 2, 4]:
            raise ValueError('Output is not divisible by {}'.format(div_fact))

        if norm_layer == 'group':
            norm_layer_func = lambda x: copy.deepcopy(nn.GroupNorm(x//2, x))
        elif norm_layer == 'batch':
            norm_layer_func = lambda x: copy.deepcopy(nn.BatchNorm2d(x))

        self.block_7 = nn.Sequential(coll.OrderedDict([
            ('map', nn.Sequential(coll.OrderedDict([
                ('refpad7', nn.ReplicationPad2d((k_size + 1) // 2)),
                ('map7', nn.Conv2d(int(512 / d_fact), 1, kernel_size=k_size*2 + 1, stride=1,
                                   padding=(k_size + 1) // 2)),
                ('sig', nn.Sigmoid()),])),
             ),

            ('conv', nn.Sequential(coll.OrderedDict([
                ('conv7', (
                    PixelRnn(int(512 / d_fact), int(128 / d_fact), rec_module=rnn_type) if pixel_rnn else
                    nn.Conv2d(int(512 / d_fact), int(512 / d_fact), kernel_size=k_size + 1, stride=1,
                              padding=(k_size + 1) // 2))
                 ),
                ('bn7', norm_layer_func(int(512 / d_fact))),
                ('relu7', nn.ReLU(inplace=True)),]))
             )
        ]))
        self.block_6 = self.build_block(int(512 / d_fact), int(512 / d_fact), k_size, norm_layer_func, 6,
                                        pixel_rnn, rnn_type)
        self.block_5 = self.build_block_up(int(512 / d_fact), int(512 / d_fact), k_size, norm_layer_func, 5)
        self.block_4 = self.build_block(int(512 / d_fact), int(512/ d_fact), k_size, norm_layer_func, 4,
                                        pixel_rnn, rnn_type)
        self.block_3 = self.build_block_up(int(512/ d_fact), int(256/ d_fact), k_size, norm_layer_func, 3)
        self.block_2 = self.build_block(int(256/ d_fact), int(128/ d_fact), k_size, norm_layer_func, 2)
        self.block_1 = self.build_block_up(int(128/ d_fact), int(64/ d_fact), k_size, norm_layer_func, 1)

        if div_fact == 2:
            self.blocks = [self.block_7, self.block_6, self.block_5, self.block_4, self.block_3, self.block_2, self.block_1]
        elif div_fact == 4:
            self.blocks = [self.block_7, self.block_6, self.block_5, self.block_4, self.block_3]
        elif div_fact == 8:
            self.blocks = [self.block_7, self.block_6, self.block_5, ]
        else:
            logger.error('Unimplemented div factor {}'.format(div_fact))
            raise NotImplementedError()

        logger.info('Final architecture is:')
        logger.info(self.blocks)

    @staticmethod
    def build_block_up(input_depth, output_depth, k_size, norm_layer_func, i):
        block = nn.Sequential(coll.OrderedDict([
            ('conv', nn.Sequential(coll.OrderedDict([
                ('conv{}'.format(i), UpConv(input_depth, input_depth, kernel_size=k_size + 1, stride=1,
                                            padding=((k_size + 1) // 2), scale_factor=2)),
                ('bn{}'.format(i), norm_layer_func(input_depth)),
                ('relu{}'.format(i), nn.ReLU(inplace=True)),
            ])),
             ),
            ('fuse', nn.Sequential(coll.OrderedDict([
                ('conv{}'.format(i), nn.Conv2d(int(input_depth * 2 + 1), output_depth, kernel_size=k_size +1,
                                                        stride=1, padding=(k_size + 1) // 2)),
                ('bn{}'.format(i), norm_layer_func(output_depth)),
                ('relu{}'.format(i), nn.ReLU(inplace=True)),
            ])),
             ),
            ('map', nn.Sequential(coll.OrderedDict([
                ('refpad{}'.format(i), nn.ReplicationPad2d((k_size + 1) // 2)),
                ('map{}'.format(i), nn.Conv2d(output_depth, 1, kernel_size=k_size + 1, stride=1)),
                ('sig', nn.Sigmoid()),
            ])),
             ),
        ]))

        return block

    @staticmethod
    def build_block(input_depth, output_depth, k_size, norm_layer_func, i, pixel_rnn=False, rnn_type='gru'):
        block = nn.Sequential(coll.OrderedDict([
            ('conv', nn.Sequential(coll.OrderedDict([
                ('conv{}'.format(i), (
                    PixelRnn(int(input_depth * 2), int(output_depth / 4), rec_module=rnn_type) if pixel_rnn else
                    nn.Conv2d(int(input_depth * 2), output_depth, kernel_size=k_size + 1,
                              stride=1, padding=(k_size + 1) // 2))
                 ),
                ('bn{}'.format(i), norm_layer_func(output_depth)),
                ('relu{}'.format(i), nn.ReLU(inplace=True)),
            ])),
             ),
        ]))

        return block

    def forward(self, unet):
        output = list()
        for i, block in enumerate(self.blocks):
            name = list(dict(block.named_children())['conv'].named_children())[0][0]
            if i == 0:
                x = dict(block.named_children())['conv'](unet[name])
                output.append(dict(block.named_children())['map'](x))
            elif 'map' in dict(block.named_children()).keys():
                x = dict(block.named_children())['conv'](x)
                upsampled_map = nn.functional.interpolate(output[-1], scale_factor=2, mode='bilinear', align_corners=True)
                x = dict(block.named_children())['fuse'](torch.cat((upsampled_map, unet[name], x), dim=1))
                output.append(dict(block.named_children())['map'](x))
            else:
                x = block(torch.cat((unet[name], x), dim=1))

        return output

    def get_training_layers(self, layers_to_train=None):
        if layers_to_train is None:
            layers_to_train = self.layers_to_train
        if layers_to_train == 'all':
            return [{'params': block.parameters()} for block in self.blocks]
        elif layers_to_train == 'no':
            return []

    def full_save(self, discard_tf=False):
        if discard_tf:
            raise NotImplementedError('Functionality not implemented')

        weights = dict()
        for i in range(7, 7 - len(self.blocks), -1):
            weights['block_{}'.format(i)] = getattr(self, 'block_{}'.format(i)).state_dict()

        return weights


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
    input_size = 448//2
    tensor_input = torch.rand([2, 3, input_size, int(input_size)])
    print(tensor_input.size())

    net = DeploymentNet()
    net(tensor_input)

    '''
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

    '''
    enc = PixEncoder(k_size=4, d_fact=2).cuda()
    #dec= PixDecoder(k_size=4, d_fact=2, out_channel=1, div_fact=2, dropout=0.1)
    dec = PixDecoderMultiscale(k_size=4, d_fact=2, div_fact=2, pixel_rnn=True).cuda()
    feat_output = enc(tensor_input)
    output = dec(feat_output)
    for out in output:
        print(out.size())

    print(dec.get_training_layers())

    rec_mod = PixelRnn(3, 1, ).cuda()

    out_rec = rec_mod(tensor_input)
    print(out_rec.size())
    '''