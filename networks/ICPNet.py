from _weakref import CallableProxyType

import torch.nn as nn
import torch
import setlog
import pose_utils.utils as utils


logger = setlog.get_logger(__name__)


class CPNet(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        self.eps = kwargs.pop('eps', 1e-5)
        self.fact = kwargs.pop('fact', 2)
        self.reject_ratio = kwargs.pop('reject_ratio', 1)
        self.layers_to_train = kwargs.pop('layers_to_train', 'all')
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        self.softmax_1d = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()


    def soft_knn(self, pc1, pc2):
        pc1_extended = pc1.new_ones(pc1.size(1), 3 * 3)
        pc2_extended = pc2.new_ones(3 * 3, pc2.size(1))

        pc1_square = pc1 ** 2
        pc1_extended[:, 2] = pc1_square[0, :]
        pc1_extended[:, 5] = pc1_square[1, :]
        pc1_extended[:, 8] = pc1_square[2, :]
        pc1_extended[:, 1] = pc1[0, :]
        pc1_extended[:, 4] = pc1[1, :]
        pc1_extended[:, 7] = pc1[2, :]

        pc2_square = pc2 ** 2
        pc2_extended[0, :] = pc2_square[0, :]
        pc2_extended[3, :] = pc2_square[1, :]
        pc2_extended[6, :] = pc2_square[2, :]
        pc2_extended[1, :] = pc2[0, :] * -2
        pc2_extended[4, :] = pc2[1, :] * -2
        pc2_extended[7, :] = pc2[2, :] * -2


        d_matrix = pc1_extended.matmul(pc2_extended)
        d_matrix = self.softmax_1d(self.fact * torch.reciprocal(d_matrix.clamp(min=self.eps)))

        pc_nearest = torch.cat([torch.sum(pc2 * prob, 1).unsqueeze(1) for i, prob in enumerate(d_matrix)], 1)
        mean_distance = torch.mean(torch.sum((pc1 - pc_nearest) ** 2, 0))

        return pc_nearest, mean_distance

    def soft_outlier_rejection(self, pc1, pc2):
        dist = torch.norm(pc1 - pc2, dim=0)
        mean_dist = torch.mean(dist, 0)
        filter = self.sigmoid((dist - mean_dist * self.reject_ratio - self.eps) * -1e10)

        return filter

    def soft_tf(self, pc1, pc2, indexor):
        pc2_centroid = torch.sum(pc2[:3, :] * indexor, -1) / torch.sum(indexor)
        pc2_centred = ((pc2[:3, :].t() - pc2_centroid).t() * indexor).t()

        pc1_centroid = torch.sum(pc1[:3, :] * indexor, -1) / torch.sum(indexor)
        pc1_centred = ((pc1[:3, :].t() - pc1_centroid).t() * indexor).t()

        H = torch.matmul(pc1_centred.t(), pc2_centred)
        logger.debug('SVD on:')
        logger.debug(H)
        U, S, V = torch.svd(H)
        if torch.det(U) * torch.det(V) < 0:
            V = V * V.new_tensor([[1, 1, -1], [1, 1, -1], [1, 1, -1]])

        R = torch.matmul(V, U.t())

        # translation
        t = pc2_centroid - torch.matmul(R, pc1_centroid)

        # homogeneous transformation
        T = pc2.new_zeros(4, 4)
        T[:3, :3] = R
        T[:3, 3] = t
        T[3, 3] = 1

        return T, utils.rot_to_quat(R), t

    def get_training_layers(self, layers_to_train=None):
        if layers_to_train is None:
            layers_to_train = self.layers_to_train
        if layers_to_train == 'all':
            return []


    def forward(self, pc1, pc2):
        '''
        Compute T from pc1 to pc2

        :param pc1: homo coord
        :param pc2: homo coord
        :return: T1->2
        '''

        T = pc1.new_zeros(pc1.size(0), 4, 4)
        q = pc1.new_zeros(pc1.size(0), 4)
        t = pc1.new_zeros(pc1.size(0), 3)

        for i, pc in enumerate(pc1):
            pc_nearest, _ = self.soft_knn(pc, pc2[i])
            indexor = self.soft_outlier_rejection(pc, pc_nearest)
            T[i], q[i], t[i] = self.soft_tf(pc, pc_nearest, indexor)

        return {'T': T, 'q': q, 't': t}


if __name__ == '__main__':
    net = CPNet()
    p1 = torch.rand(2, 4, 5000)
    p1[:, 3, :] = 1
    p2 = p1 + 0.001
    p2[:, 3, :] = 1

    T = net(p1, p2)
    print(T[0])
    print(T[0].matmul(p1))
    print(p2 - T[0].matmul(p1))

