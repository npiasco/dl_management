import torch.nn as nn
import torch.nn.functional as nnf
import torch
import setlog
import pose_utils.utils as utils


logger = setlog.get_logger(__name__)


class STNkD(nn.Module):
    """
    Spatial Transformer Net for PointNet, producing a KxK transformation matrix.
    Parameters:
      nfeat: number of input features
      nf_conv: list of layer widths of point embeddings (before maxpool)
      nf_fc: list of layer widths of joint embeddings (after maxpool)
    """
    def __init__(self, nfeat, nf_conv, nf_fc, K=2):
        nn.Module.__init__(self)

        modules = []
        for i in range(len(nf_conv)):
            modules.append(nn.Conv1d(nf_conv[i-1] if i>0 else nfeat, nf_conv[i], 1))
            modules.append(nn.BatchNorm1d(nf_conv[i]))
            modules.append(nn.ReLU(True))
        self.convs = nn.Sequential(*modules)

        modules = []
        for i in range(len(nf_fc)):
            modules.append(nn.Linear(nf_fc[i-1] if i>0 else nf_conv[-1], nf_fc[i]))
            modules.append(nn.BatchNorm1d(nf_fc[i]))
            modules.append(nn.ReLU(True))
        self.fcs = nn.Sequential(*modules)

        self.proj = nn.Linear(nf_fc[-1], K*K)
        nn.init.constant(self.proj.weight, 0); nn.init.constant(self.proj.bias, 0)
        self.eye = torch.eye(K).unsqueeze(0)

    def forward(self, input):
        self.eye = self.eye.cuda() if input.is_cuda else self.eye
        input = self.convs(input)
        input = nnf.max_pool1d(input, input.size(2)).squeeze(2)
        input = self.fcs(input)
        input = self.proj(input)
        return input.view(-1,self.eye.size(1),self.eye.size(2)) + self.eye

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
    def __init__(self, nf_conv, nf_fc, nf_conv_stn, nf_fc_stn, nfeat, nfeat_stn=2, nfeat_global=1, prelast_do=0.5, last_ac=False):
        super(PointNet, self).__init__()
        if nfeat_stn > 0:
            self.stn = STNkD(nfeat_stn, nf_conv_stn, nf_fc_stn)
        self.nfeat_stn = nfeat_stn

        modules = []
        for i in range(len(nf_conv)):
            modules.append(nn.Conv1d(nf_conv[i-1] if i>0 else nfeat, nf_conv[i], 1))
            modules.append(nn.BatchNorm1d(nf_conv[i]))
            modules.append(nn.ReLU(True))
        self.convs = nn.Sequential(*modules)

        modules = []
        for i in range(len(nf_fc)):
            modules.append(nn.Linear(nf_fc[i-1] if i>0 else nf_conv[-1]+nfeat_global, nf_fc[i]))
            if i<len(nf_fc)-1 or last_ac:
                modules.append(nn.BatchNorm1d(nf_fc[i]))
                modules.append(nn.ReLU(True))
            if i==len(nf_fc)-2 and prelast_do>0:
                modules.append(nn.Dropout(prelast_do))
        self.fcs = nn.Sequential(*modules)

    def forward(self, input, input_global):
        if self.nfeat_stn > 0:
            T = self.stn(input[:,:self.nfeat_stn,:])
            xy_transf = torch.bmm(input[:,:2,:].transpose(1,2), T).transpose(1,2)
            input = torch.cat([xy_transf, input[:,2:,:]], 1)

        input = self.convs(input)
        input = nnf.max_pool1d(input, input.size(2)).squeeze(2)
        if input_global is not None:
            input = torch.cat([input, input_global.view(-1,1)], 1)
        return self.fcs(input)


if __name__ == '__main__':

    torch.manual_seed(10)
    device = 'cpu'
    batch = 1
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