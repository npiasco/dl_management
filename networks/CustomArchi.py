import setlog
import torch.nn as nn
import torch.autograd as auto
import torch
import networks.Descriptor as Desc
import networks.ResNet as ResNet
import networks.FeatAggregation as Agg


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

if __name__ == '__main__':
    tensor_input = torch.rand([10, 3, 224, 224])
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

    feat_output = net(auto.Variable(tensor_input))
    print(feat_output)
