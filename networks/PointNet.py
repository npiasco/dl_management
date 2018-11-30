import torch.nn as nn
import torch.nn.functional as nnf
import torch
import setlog
import pose_utils.utils as utils
import copy


logger = setlog.get_logger(__name__)


class STNkD(nn.Module):
    """
    Spatial Transformer Net for PointNet, producing a KxK transformation matrix.
    Parameters:
      nfeat: number of input features
      nf_conv: list of layer widths of point embeddings (before maxpool)
      nf_fc: list of layer widths of joint embeddings (after maxpool)
    """
    def __init__(self, nfeat, nf_conv, nf_fc, **kwargs):
        nn.Module.__init__(self)
        K = kwargs.pop('K', 3)
        norm_layer = kwargs.pop('norm_layer', 'group')
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if norm_layer == 'group':
            norm_layer_func = lambda x: copy.deepcopy(nn.GroupNorm(x//2, x))
        elif norm_layer == 'batch':
            norm_layer_func = lambda x: copy.deepcopy(nn.BatchNorm1d(x))

        modules = []
        for i in range(len(nf_conv)):
            modules.append(nn.Conv1d(nf_conv[i-1] if i>0 else nfeat, nf_conv[i], 1))
            modules.append(norm_layer_func(nf_conv[i]))
            modules.append(nn.ReLU(True))
        self.convs = nn.Sequential(*modules)

        modules = []
        for i in range(len(nf_fc)):
            modules.append(nn.Linear(nf_fc[i-1] if i>0 else nf_conv[-1], nf_fc[i]))
            modules.append(norm_layer_func(nf_fc[i]))
            modules.append(nn.ReLU(True))
        self.fcs = nn.Sequential(*modules)

        self.proj = nn.Linear(nf_fc[-1], K*K)
        nn.init.constant_(self.proj.weight, 0)
        nn.init.constant_(self.proj.bias, 0)
        self.eye = nn.Parameter(torch.eye(K).unsqueeze(0), requires_grad=False)

    def forward(self, input):
        input = self.convs(input)
        input = nnf.max_pool1d(input, input.size(2)).squeeze(2)
        input = self.fcs(input)
        input = self.proj(input)
        return input.view(-1, self.eye.size(1), self.eye.size(2)) + self.eye

class PointNet(nn.Module):
    """
    PointNet with only one spatial transformer and additional "global" input concatenated after maxpool.
    Parameters:
      nf_conv: list of layer widths of point embeddings (before maxpool)
      nf_fc: list of layer widths of joint embeddings (after maxpool)
      nfeat: number of input features
      nf_conv_stn, nf_fc_stn, nfeat_stn: as above but for Spatial transformer
      nfeat_global: number of features concatenated after maxpooling
      prelast_do: dropout after the pre-last parameteric layer
      last_ac: whether to use batch norm and relu after the last parameteric layer
    """
    def __init__(self, nf_conv, nf_fc, nf_conv_stn, nf_fc_stn, nf_conv_desc, nfeat, **kwargs):
        nn.Module.__init__(self)

        nfeat_stn = kwargs.pop('nfeat_stn',3)
        nfeat_global = kwargs.pop('nfeat_global', 0)
        prelast_do = kwargs.pop('prelast_do', 0.5)
        last_ac = kwargs.pop('last_ac', False)
        self.normalize_p = kwargs.pop('normalize_p', True)
        self.normalize_f = kwargs.pop('normalize_f', True)
        self.layers_to_train = kwargs.pop('layers_to_train', 'all')
        norm_layer = kwargs.pop('norm_layer', 'group')
        end_relu = kwargs.pop('end_relu', False)

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if norm_layer == 'group':
            norm_layer_func = lambda x: copy.deepcopy(nn.GroupNorm(x//2, x))
        elif norm_layer == 'batch':
            norm_layer_func = lambda x: copy.deepcopy(nn.BatchNorm1d(x))

        if nfeat_stn > 0:
            self.stn = STNkD(nfeat_stn, nf_conv_stn, nf_fc_stn)
        self.nfeat_stn = nfeat_stn

        modules = []
        for i in range(len(nf_conv)):
            modules.append(nn.Conv1d(nf_conv[i-1] if i>0 else nfeat, nf_conv[i], 1))
            modules.append(norm_layer_func(nf_conv[i]))

            modules.append(nn.ReLU(True))
        self.convs = nn.Sequential(*modules)

        modules = []
        for i in range(len(nf_fc)):
            modules.append(nn.Linear(nf_fc[i-1] if i>0 else nf_conv[-1]+nfeat_global, nf_fc[i]))
            if i<len(nf_fc)-1 or last_ac:
                modules.append(norm_layer_func(nf_fc[i]))
                modules.append(nn.ReLU(True))
            if i==len(nf_fc)-2 and prelast_do>0:
                modules.append(nn.Dropout(prelast_do))
        self.fcs = nn.Sequential(*modules)

        modules = []
        for i in range(len(nf_conv_desc)):
            modules.append(nn.Conv1d(nf_conv_desc[i - 1] if i > 0 else nf_conv[-1] + nf_fc[-1], nf_conv_desc[i], 1))
            modules.append(norm_layer_func(nf_conv_desc[i]))
            modules.append(nn.ReLU(True))

        if not end_relu:
            modules.pop()

        self.convs_desc = nn.Sequential(*modules)

    def forward(self, input_p, input_f, input_global=None):
        if self.normalize_p:
            centroid = torch.mean(input_p, -1).unsqueeze(-1)
            input_p = input_p - centroid

        if self.normalize_f:
            input_f = nnf.normalize(input_f, dim=1)

        if self.nfeat_stn > 0:
            T = self.stn(input_p[:, :3, :])
            #xy_transf = torch.bmm(input[:,:3,:].transpose(1,2), T).transpose(1,2)
            xy_transf = T.matmul(input_p[:, :3, :])
            input = torch.cat([xy_transf, input_f], 1)
        else:
            input = torch.cat([input_p[:, :3, :], input_f], 1)

        input = self.convs(input)
        max_input = nnf.max_pool1d(input, input.size(2)).squeeze(2)
        if input_global is not None:
            max_input = torch.cat([max_input, input_global.view(-1,1)], 1)
        global_desc = self.fcs(max_input).unsqueeze(-1)
        # in the original paper, concatenation is done on max_input & input
        input = torch.cat((input, global_desc.repeat(1, 1, input.size(-1))), 1)
        return self.convs_desc(input)

    def get_training_layers(self, layers_to_train=None):
        if layers_to_train is None:
            layers_to_train = self.layers_to_train
        if layers_to_train == 'all':
            return [{'params': self.parameters()}]

if __name__ == '__main__':

    torch.manual_seed(10)
    device = 'cpu'
    batch = 2
    nb_pt = 5000
    p1 = torch.rand(batch, 4, nb_pt).to(device)
    p1[:, 3, :] = 1
    T = torch.zeros(4, 4).to(device)
    T[:3, :3] = utils.rotation_matrix(torch.tensor([1.0, 0, 0]), torch.tensor([0.05]))
    T[:3, 3] = torch.tensor([0.1, 0, 0])
    T[3, 3] = 1
    p2 = T.matmul(p1)
    #p2 = torch.rand(2, 4, nb_pt).to(device)
    desc1 = torch.rand(batch, 16, nb_pt).to(device)
    desc2 = desc1

    net = PointNet(nf_conv=[64,64,128,128,256],
                   nf_fc=[256,64,32],
                   nf_conv_stn=[64, 64, 128],
                   nf_fc_stn=[128, 64],
                   nf_conv_desc=[64, 64, 4],
                   nfeat=3+16, nfeat_global=0, nfeat_stn=0, end_relu=False)
    print(net)
    print(net(p1, desc1).size())