import setlog
import torch.nn as nn
import torch
import networks.Descriptor as Desc
import networks.ResNet as ResNet
import networks.FeatAggregation as Agg
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
                'unet': True
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

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        base_archi = [
            ('conv0', nn.Conv2d(i_channel, int(64 / d_fact), kernel_size=k_size, stride=2)),
            ('relu0', nn.LeakyReLU(0.2, inplace=True)),
            ('conv1', nn.Conv2d(int(64 / d_fact), int(128 / d_fact), kernel_size=k_size+1, stride=1, padding=1)),
            ('bn1', nn.BatchNorm2d(int(128/d_fact))),
            ('relu1', nn.LeakyReLU(0.2, inplace=True)),
            ('conv2', nn.Conv2d(int(128 / d_fact), int(256 / d_fact), kernel_size=k_size, stride=2)),
            ('bn2', nn.BatchNorm2d(int(256 / d_fact))),
            ('relu2', nn.LeakyReLU(0.2, inplace=True)),
            ('conv3', nn.Conv2d(int(256 / d_fact), int(512 / d_fact), kernel_size=k_size+1, stride=1, padding=1)),
            ('bn3', nn.BatchNorm2d(int(512 / d_fact))),
            ('relu3', nn.LeakyReLU(0.2, inplace=True)),
            ('conv4', nn.Conv2d(int(512 / d_fact), int(512 / d_fact), kernel_size=k_size, stride=2)),
            ('bn4', nn.BatchNorm2d(int(512 / d_fact))),
            ('relu4', nn.LeakyReLU(0.2, inplace=True)),
            ('conv5', nn.Conv2d(int(512 / d_fact), int(512 / d_fact), kernel_size=k_size+1, stride=1, padding=1)),
            ('bn5', nn.BatchNorm2d(int(512 / d_fact))),
            ('relu5', nn.LeakyReLU(0.2, inplace=True)),
            ('conv6', nn.Conv2d(int(512 / d_fact), int(512 / d_fact), kernel_size=k_size, stride=2)),
            ('bn6', nn.BatchNorm2d(int(512 / d_fact))),
            ('relu6', nn.LeakyReLU(0.2, inplace=True)),
            ('conv7', nn.Conv2d(int(512 / d_fact), int(512 / d_fact), kernel_size=k_size+1, stride=1, padding=1)),
            ('bn7', nn.BatchNorm2d(int(512 / d_fact))),
            ('relu7', nn.LeakyReLU(0.2, inplace=True)),
        ]

        self.feature = nn.Sequential(
            coll.OrderedDict(base_archi)
        )

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


class PixDecoder(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        self.layers_to_train = kwargs.pop('layers_to_train', 'all')
        out_channel = kwargs.pop('out_channel', 1)
        d_fact = kwargs.pop('d_fact', 1)
        k_size = kwargs.pop('k_size', 2)
        out_pad = kwargs.pop('out_pad', 0)
        #TODO: construct various archi depending on the desired scale factor according to the output size

        base_archi = [
            ('conv7', nn.ConvTranspose2d(int(512 / d_fact), int(512 / d_fact), kernel_size=k_size+1, stride=1, padding=1)),
            ('relu7', nn.ReLU(inplace=True)),
            ('conv6', nn.ConvTranspose2d(int(512 / d_fact * 2), int(512 / d_fact), kernel_size=k_size, stride=2, output_padding=out_pad)),
            ('bn6', nn.BatchNorm2d(int(512 / d_fact))),
            ('relu6', nn.ReLU(inplace=True)),
            ('conv5', nn.ConvTranspose2d(int(512 / d_fact * 2), int(512 / d_fact), kernel_size=k_size+1, stride=1, padding=1)),
            ('bn5', nn.BatchNorm2d(int(512 / d_fact))),
            ('relu5', nn.ReLU(inplace=True)),
            ('conv4', nn.ConvTranspose2d(int(512 / d_fact * 2), int(512 / d_fact), kernel_size=k_size, stride=2, output_padding=0)),
            ('bn4', nn.BatchNorm2d(int(512 / d_fact))),
            ('relu4', nn.ReLU(inplace=True)),
            ('conv3', nn.ConvTranspose2d(int(512 / d_fact * 2), int(256 / d_fact), kernel_size=k_size+1, stride=1, padding=1)),
            ('bn3', nn.BatchNorm2d(int(256 / d_fact))),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv2', nn.ConvTranspose2d(int(256 / d_fact * 2), int(128 / d_fact), kernel_size=k_size, stride=2, output_padding=0)),
            ('bn2', nn.BatchNorm2d(int(128 / d_fact))),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv1', nn.ConvTranspose2d(int(128 / d_fact * 2), int(64 / d_fact), kernel_size=k_size+1, stride=1, padding=1)),
            ('bn1', nn.BatchNorm2d(int(64 / d_fact))),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv0', nn.ConvTranspose2d(int(64 / d_fact * 2), int(out_channel), kernel_size=k_size, stride=2)),
            ('sig0', nn.Sigmoid()),
        ]

        self.feature = nn.Sequential(
            coll.OrderedDict(base_archi)
        )

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



if __name__ == '__main__':
    input_size = 224
    tensor_input = torch.rand([10, 3, input_size, input_size])
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
    enc = PixEncoder(k_size=2, d_fact=4)
    dec= PixDecoder(k_size=2, d_fact=4, out_channel=1)

    feat_output = enc(tensor_input)
    output = dec(feat_output)
    print(output.size())
    #print([res.size() for res in feat_output.values()])

