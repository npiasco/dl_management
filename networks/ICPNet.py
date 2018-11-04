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

    def directtf(self, pc1, pc2, indexor):
        _, indices = torch.unique(pc2[0, :], return_inverse=True)
        unique_indices = torch.unique(indices)
        #M = pc1.new_zeros(indexor.nonzero().numel() * 2, 12)
        A = pc1.new_zeros(indexor[unique_indices].nonzero().numel() * 3, 12)
        B = pc1.new_zeros(indexor[unique_indices].nonzero().numel() * 3, 1)

        if indexor.nonzero().numel() < 3:
            logger.warn('Not enought correspondances to use dlt ({} match)'.format(
                indexor.nonzero().numel()))
        cpt = 0
        for i in unique_indices:
            if indexor[i].item():
                A[cpt, :4] = pc1[:, i]
                A[cpt + 1, 4:8] = pc1[:, i]
                A[cpt + 2, 8:] = pc1[:, i]
                B[cpt] = pc2[0, i]
                B[cpt + 1] = pc2[1, i]
                B[cpt + 2] = pc2[2, i]

                cpt += 3

        X, _ = torch.gels(B, A)
        X = X[:12]

        P = X.view(3, 4)
        T = pc2.new_zeros(4, 4)
        T[:3, :] = P
        T[3, 3] = 1
        #R = T[:3, :3]
        #R = utils.quat_to_rot(utils.rot_to_quat(T[:3, :3]))

        R = normalize_rotmat(T[:3, :3])
        if torch.det(R) < 0:
            print('Inverse')
            R *= -1

        T[:3, :3] = R

        return T, utils.rot_to_quat(T[:3, :3]), T[:3, 3]

    def dlt(self, pc1, pc2, indexor):

        std, mean = torch.std(pc2[:3, :], 1), torch.mean(pc2[:3, :], 1)
        T2std, T2mean = pc2.new_zeros(4, 4), pc2.new_zeros(4, 4)
        T2mean[0, 0] = T2mean[1, 1] = T2mean[2, 2] = T2mean[3, 3] = 1
        T2mean[:3, 3] = -mean
        T2std[3, 3] = 1
        for i in range(3):
            T2std[i, i] = 1/std[i]
        T2 = torch.matmul(T2std, T2mean)

        pc2 = T2.matmul(pc2)

        std, mean = torch.std(pc1[:3, :], 1), torch.mean(pc1[:3, :], 1)
        T1std, T1mean = pc1.new_zeros(4, 4), pc1.new_zeros(4, 4)
        T1mean[0, 0] = T1mean[1, 1] = T1mean[2, 2] = T1mean[3, 3] = 1
        T1mean[:3, 3] = -mean
        T1std[3, 3] = 1
        for i in range(3):
            T1std[i, i] = 1/std[i]
        T1 = torch.matmul(T1std, T1mean)

        pc1 = T1.matmul(pc1)

        _, indices = torch.unique(pc2[0, :], return_inverse=True)
        unique_indices = torch.unique(indices)
        #M = pc1.new_zeros(indexor.nonzero().numel() * 2, 12)
        M = pc1.new_zeros(indexor[unique_indices].nonzero().numel() * 2, 12)
        print('Processing {} unique matches'.format(unique_indices.numel()))
        if indexor[unique_indices].nonzero().numel() < 6:
            logger.warn('Not enought correspondances to use dlt ({} match)'.format(indexor[unique_indices].nonzero().numel()))
        cpt = 0
        for i in unique_indices:
            if indexor[i].item():
                M[cpt, 4:] = torch.cat((-pc2[2, i]*pc1[:, i], pc2[1, i]*pc1[:, i]), 0)
                M[cpt + 1, :4] = pc2[2, i]*pc1[:, i]
                M[cpt + 1, 8:] = -pc2[0, i]*pc1[:, i]
#                M[cpt + 2, :8] = torch.cat((-pt[1]*pc12[:, i], pt[0]*pc1[:, i]), 0)
                cpt += 2

        U, S, V = torch.svd(M)
        p = V[:, -1]

        if p[10].item() < 0:  # Diag of rot mat should be > 0
            print('inverse')
            p = p * -1

        norm = (p[8] ** 2 + p[9] ** 2 + p[10] ** 2) ** 0.5
        p = p / norm
        P = p.view(3, 4)

        # homogeneous transformation
        T = pc2.new_zeros(4, 4)
        T[:3, :] = P
        T[3, 3] = 1
        #T = T1.inverse().matmul(T.matmul(T2))
        T = T2.inverse().matmul(T.matmul(T1))
        print(T[:3, :3].matmul(T[:3, :3].t()))
        T[:3, :3] = normalize_rotmat(T[:3, :3])
        return T, utils.rot_to_quat(T[:3, :3]), T[:3, 3]

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
            #T[i], q[i], t[i] = self.soft_tf(pc, pc_nearest, indexor)
            T[i], q[i], t[i] = self.dlt(pc, pc_nearest, indexor)
            #print(T[i])
            #T[i], q[i], t[i] = self.directtf(pc, pc_nearest, indexor)
            print(T[i])

        return {'T': T, 'q': q, 'p': t}

def normalize_rotmat(R):
    z = R[:, 2]/torch.norm(R[:, 2])
    y = R[:, 1]
    x = torch.cross(y, z)
    x = x/torch.norm(x)
    y = torch.cross(z, x)
    R[:, 0] = x
    R[:, 1] = y
    R[:, 2] = z
    return R

if __name__ == '__main__':
    net = CPNet(fact=2000)
    p1 = torch.rand(1, 4, 40)
    p1[:, 3, :] = 1
    T = torch.zeros(4, 4)
    T[:3, :3] = utils.rotation_matrix(torch.tensor([1.0, 0, 0]), torch.tensor([0.05]))
    T[:3, 3] = torch.tensor([0.1, 0, 0])
    T[3, 3] = 1
    print(T)
    p2 = T.matmul(p1)
    """
    p1 + 0.001
    p2[:, 3, :] = 1
    """
    T = net(p1, p2)
    print(T['T'])
    """
    print(T['T'][0].matmul(p1))
    print(p2 - T['T'][0].matmul(p1))
    """

