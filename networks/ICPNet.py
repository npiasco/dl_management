import torch.nn as nn
import torch.nn.functional as func_nn
import torch
import setlog
import pose_utils.utils as utils
import sklearn.neighbors as neighbors
import scipy.spatial.distance as dst_func
import time
import numpy as np


logger = setlog.get_logger(__name__)


class CPNet(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        self.eps = kwargs.pop('eps', 1e-5)
        self.fact = kwargs.pop('fact', 2)
        self.reject_ratio = kwargs.pop('reject_ratio', 1)
        self.layers_to_train = kwargs.pop('layers_to_train', 'all')
        self.outlier_filter = kwargs.pop('outlier_filter', True)
        self.use_dst_pt = kwargs.pop('use_dst_pt', True)
        self.use_dst_desc = kwargs.pop('use_dst_desc', False)
        self.normalize_desc = kwargs.pop('normalize_desc', True)
        self.desc_p = kwargs.pop('desc_p', 2)
        self.pose_solver = kwargs.pop('pose_solver', 'svd')
        knn = kwargs.pop('knn', 'soft')

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if self.pose_solver not in ('svd', 'eig', 'dlt', 'lsq'):
            raise AttributeError('No pose solver named {}'.format(self.pose_solver))

        self.softmax_1d = nn.Softmax(dim=1)
        self.softmax_0d = nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid()
        self.pdist = torch.nn.PairwiseDistance(p=1, keepdim=True)
        self.is_hard_knn = False
        if knn == 'soft':
            self.knn = self.soft_knn
        elif knn == 'hard':
            self.knn = self.hard_knn
            self.is_hard_knn = True
        else:
            raise('Unknown knn type {}'.format(knn))

    def spatial_distance(self, pc1, pc2):
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

        return d_matrix

    def descriptor_distance(self, d1, d2):
        '''
        :param d1: desc of pc1, [s_desc, s_pc1]
        :param d2:
        :return:
        '''

        if self.normalize_desc:
            d1 = func_nn.normalize(d1, dim=0)
            d2 = func_nn.normalize(d2, dim=0)

        d_matrix = torch.pow(d1.t().matmul(d2), self.desc_p)

        return d_matrix

    def hard_knn(self, pc1, pc2, *args):
        distance = 1
        if self.use_dst_pt:
            distance *= torch.reciprocal(self.spatial_distance(pc1, pc2).clamp(min=self.eps))
        if self.use_dst_desc:
            if args:
                distance *= self.descriptor_distance(args[0], args[1])

        d_matrix = self.softmax_1d(self.fact * distance)
        d_matrix_bi = self.softmax_0d(self.fact * distance)
        pc_nearest = pc1.clone()
        indexor = pc1.new_zeros(pc1.size(1))
        for i, pt in enumerate(pc1.t()):
            idx_1 = torch.argmax(d_matrix[i, :], 0)
            idx_2 = torch.argmax(d_matrix_bi[:, idx_1], 0)
            if i == idx_2.item():
                pc_nearest[:, i] = pc2[:, idx_1].clone()
                indexor[i] = 1

        mean_distance = torch.mean(func_nn.pairwise_distance(pc1.t(), pc_nearest.t(), p=2))

        return pc_nearest, mean_distance, indexor

    def soft_knn(self, pc1, pc2, *args):

        distance = 1
        if self.use_dst_pt:
            distance *= torch.reciprocal(self.spatial_distance(pc1, pc2).clamp(min=self.eps))
        if self.use_dst_desc:
            if args:
                distance *= self.descriptor_distance(args[0], args[1])
        d_matrix = self.softmax_1d(self.fact * distance)

        pc_nearest = torch.cat([torch.sum(pc2 * prob, 1).unsqueeze(1) for i, prob in enumerate(d_matrix)], 1)

        mean_distance = torch.mean(func_nn.pairwise_distance(pc1.t(), pc_nearest.t(), p=2))

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

    def quat_tf(self, pc1, pc2, indexor):
        pc2_centroid = torch.sum(pc2[:3, :] * indexor, -1) / torch.sum(indexor)
        pc2_centred = ((pc2[:3, :].t() - pc2_centroid).t() * indexor).t()

        pc1_centroid = torch.sum(pc1[:3, :] * indexor, -1) / torch.sum(indexor)
        pc1_centred = ((pc1[:3, :].t() - pc1_centroid).t() * indexor).t()

        S = torch.matmul(pc1_centred.t(), pc2_centred) / pc1.size(1)
        s = pc1_centred.new_tensor([S[1, 2] -  S[2, 1], S[2, 0] -  S[0, 2], S[0, 1] -  S[1, 0]])
        eye_mat = pc1_centred.new_zeros(3, 3)
        eye_mat[0, 0] = eye_mat[1, 1] = eye_mat[2, 2] = 1
        W = pc1_centred.new_zeros(4, 4)
        W[0, 0] = torch.trace(S)
        W[0, 1:] = s
        W[1:, 0] = s
        W[1:, 1:] = S + S.t() - torch.trace(S)*eye_mat

        #e, v = torch.eig(W, eigenvectors=True)
        e, v = torch.symeig(W, eigenvectors=True)

        q = v.t()[torch.argmax(e)]
        #q = v[torch.argmax(e[:, 0])]
        #q = q.new_tensor([q[3], q[0], q[1], q[2]])

        R = utils.quat_to_rot(q)

        # translation
        t = pc2_centroid - torch.matmul(R, pc1_centroid)

        # homogeneous transformation
        T = pc2.new_zeros(4, 4)
        T[:3, :3] = R
        T[:3, 3] = t
        T[3, 3] = 1

        return T, q, t

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


    def forward(self, pc1, pc2, *args):
        '''
        Compute T from pc1 to pc2

        :param pc1: homo coord
        :param pc2: homo coord
        :return: T1->2
        '''

        T = pc1.new_zeros(pc1.size(0), 4, 4)
        q = pc1.new_zeros(pc1.size(0), 4)
        t = pc1.new_zeros(pc1.size(0), 3)
        errors = pc1.new_zeros(pc1.size(0), 1)

        for i, pc in enumerate(pc1):
            if args:
                if self.is_hard_knn:
                    pc_nearest, errors[i], indexor = self.knn(pc, pc2[i], args[0][i], args[1][i])
                else:
                    pc_nearest, errors[i] = self.knn(pc, pc2[i], args[0][i], args[1][i])
            else:
                if self.is_hard_knn:
                    pc_nearest, errors[i], indexor = self.knn(pc, pc2[i])
                else:
                    pc_nearest, errors[i] = self.knn(pc, pc2[i])

            if not self.is_hard_knn:
                if self.outlier_filter:
                    indexor = self.soft_outlier_rejection(pc, pc_nearest)
                else:
                    indexor = pc_nearest.new_ones(pc_nearest.size(1))
            if self.pose_solver == 'svd':
                T[i], q[i], t[i] = self.soft_tf(pc, pc_nearest, indexor)
            elif self.pose_solver == 'eig':
                T[i], q[i], t[i] = self.quat_tf(pc, pc_nearest, indexor)
            elif self.pose_solver == 'dlt':
                T[i], q[i], t[i] = self.dlt(pc, pc_nearest, indexor)
            elif self.pose_solver == 'lsq':
                T[i], q[i], t[i] = self.directtf(pc, pc_nearest, indexor)

        return {'T': T, 'q': q, 'p': t, 'err': errors}


class MatchNet(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        self.eps = kwargs.pop('eps', 1e-5)
        self.fact = kwargs.pop('fact', 2)
        self.reject_ratio = kwargs.pop('reject_ratio', 1)
        self.desc_p = kwargs.pop('desc_p', 2)
        self.matching_ratio = kwargs.pop('matching_ratio', 0.9)
        self.layers_to_train = kwargs.pop('layers_to_train', 'all')
        self.outlier_filter = kwargs.pop('outlier_filter', True)
        self.use_dst_pt = kwargs.pop('use_dst_pt', True)
        self.use_dst_desc = kwargs.pop('use_dst_desc', False)
        self.normalize_desc = kwargs.pop('normalize_desc', True)
        self.bidirectional = kwargs.pop('bidirectional', False)
        self.n_neighbors = kwargs.pop('n_neighbors', 10)
        self.nn_ratio = kwargs.pop('nn_ratio', 0.1)
        knn = kwargs.pop('knn', 'soft')
        knn_metric = kwargs.pop('knn_metric', 'minkowski')

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        self.hard = False
        if knn == 'soft':
            self.knn = self.soft_knn
        elif knn == 'hard':
            self.knn = self.hard_knn
            self.hard = True
            if self.bidirectional:
                self.outlier_filter = False
        elif knn == 'hard_cpu':
            self.knn = self.hard_knn_cpu
            self.hard = True
            if self.bidirectional:
                self.outlier_filter = False
            if self.use_dst_desc and not self.use_dst_pt and self.normalize_desc:
                self.nn_computor = neighbors.NearestNeighbors(n_neighbors=1, metric='cosine')
            else:
                self.nn_computor = neighbors.NearestNeighbors(n_neighbors=1)
                                                          #metric=self.custom_metric)
        elif knn == 'fast_soft_knn':
            self.knn = self.fast_soft_knn
            self.outlier_filter = False
            self.nn_computor = neighbors.NearestNeighbors(n_neighbors=self.n_neighbors, metric=knn_metric)
        elif knn == 'bidirectional':
            self.hard = True
            self.knn = self.bidirectional_matching
            self.outlier_filter = False
            self.nn_computor = neighbors.NearestNeighbors(n_neighbors=1, metric=knn_metric)
        elif knn == 'ratio':
            self.hard = True
            self.knn = self.matching_w_ratio
            self.outlier_filter = False
            self.nn_computor = neighbors.NearestNeighbors(n_neighbors=2, metric=knn_metric)
        elif knn == 'nearest_match':
            self.hard = True
            self.knn = self.nearest_match
            self.outlier_filter = False
            self.nn_computor = neighbors.NearestNeighbors(n_neighbors=50, metric='euclidean')
        elif knn == 'desc_reweighting':
            self.knn = self.desc_reweighting
            self.outlier_filter = False
            self.nn_computor = neighbors.NearestNeighbors(n_neighbors=self.n_neighbors, metric=knn_metric)
        else:
            raise NotImplementedError('Unknown knn type {}'.format(knn))

        self.softmax_1d = nn.Softmax(dim=1)
        self.softmax_0d = nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid()
        self.pdist = torch.nn.PairwiseDistance(p=1, keepdim=True)
        self.fitted = False

    def fit(self, data):
        if self.fitted == False:
            self.fitted = True
            data = data.view(-1, data.size(-1))
            if self.normalize_desc:
                data = func_nn.normalize(data, dim=0)
            data = data.detach().t().cpu().numpy()

            self.nn_computor.fit(data)
        else:
            logger.debug('Trying to fit a second time the nn machine')

    def unfit(self):
        self.fitted = False

    @staticmethod
    def custom_metric(u, v):
        #return dst_func.minkowski(u[:3], v[:3], p=2)*dst_func.cosine(u[3:], v[3:])
        return sum((u[:3] - v[:3])**2) * 1/np.dot(u[3:], v[3:])

    def spatial_distance(self, pc1, pc2):
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

        return d_matrix

    def descriptor_distance(self, d1, d2):
        '''
        :param d1: desc of pc1, [s_desc, s_pc1]
        :param d2:
        :return:
        '''
        # Desc flattening
        d1 = d1.view(d1.size(0), -1)
        d2 = d2.view(d2.size(0), -1)
        if self.normalize_desc:
            d1 = func_nn.normalize(d1, dim=0)
            d2 = func_nn.normalize(d2, dim=0)

        d_matrix = torch.pow(d1.t().matmul(d2), self.desc_p)

        return d_matrix

    def hard_knn_cpu(self, pc1, pc2, *args):
        pc_nearest = pc1.clone()
        if self.bidirectional:
            indexor = pc1.new_zeros(pc1.size(1))
        else:
            indexor = pc1.new_ones(pc1.size(1))

        pc1_cpu = pc1[:3, :].detach().t().cpu().numpy()
        pc2_cpu = pc2[:3, :].detach().t().cpu().numpy()

        if self.use_dst_desc:
            if args:
                d1 = args[0].view(args[0].size(0), -1)
                d2 = args[1].view(args[1].size(0), -1)
                if self.normalize_desc:
                    d1 = func_nn.normalize(d1, dim=0)
                    d2 = func_nn.normalize(d2, dim=0)
                d1_cpu = d1.detach().t().cpu().numpy()
                d2_cpu = d2.detach().t().cpu().numpy()
                if self.use_dst_pt:
                    pc1_cpu = np.concatenate((pc1_cpu, d1_cpu), axis=1)
                    pc2_cpu = np.concatenate((pc2_cpu, d2_cpu), axis=1)
                else:
                    pc1_cpu = d1_cpu
                    pc2_cpu = d2_cpu

        self.nn_computor.fit(pc2_cpu)
        idx_nn_2 = self.nn_computor.kneighbors(pc1_cpu, return_distance=False)
        if self.bidirectional:
            self.nn_computor.fit(pc1_cpu)
            idx_nn_1 = self.nn_computor.kneighbors(pc2_cpu, return_distance=False)

        for i in range(pc1.size(1)):
            idx_1 = idx_nn_2[i][0]
            if self.bidirectional:
                idx_2 = idx_nn_1[idx_1][0]
                if i == idx_2:
                    pc_nearest[:, i] = pc2[:, idx_1]
                    indexor[i] = 1
            else:
                pc_nearest[:, i] = pc2[:, idx_1]

        return pc_nearest, indexor

    def hard_knn(self, pc1, pc2, *args):
        '''
        Deprecated func
        '''
        logger.warning('Deprectated method.')
        distance = 1
        if self.use_dst_pt:
            distance *= torch.reciprocal(self.spatial_distance(pc1, pc2).clamp(min=self.eps))
        if self.use_dst_desc:
            if args:
                distance *= self.descriptor_distance(args[0], args[1])

        d_matrix = self.softmax_1d(self.fact * distance)
        if self.bidirectional:
            d_matrix_bi = self.softmax_0d(self.fact * distance)
        pc_nearest = pc1.clone()
        if self.bidirectional:
            indexor = pc1.new_zeros(pc1.size(1))
        else:
            indexor = pc1.new_ones(pc1.size(1))

        for i in range(pc1.size(1)):
            idx_1 = torch.argmax(d_matrix[i, :], 0)
            if self.bidirectional:
                idx_2 = torch.argmax(d_matrix_bi[:, idx_1], 0)
                if i == idx_2.item():
                    pc_nearest[:, i] = pc2[:, idx_1]
                    indexor[i] = 1
            else:
                pc_nearest[:, i] = pc2[:, idx_1]

        return pc_nearest, indexor


    def matching_w_ratio(self, pc1, pc2, desc1, desc2):
        d1 = desc1.view(desc1.size(0), -1)
        d2 = desc2.view(desc2.size(0), -1)
        if self.normalize_desc:
            d1 = func_nn.normalize(d1, dim=0)
            d2 = func_nn.normalize(d2, dim=0)
        d1_cpu = d1.detach().t().cpu().numpy()

        if not self.fitted:
            self.fit(desc2)
            self.fitted = False

        distances, idx_nn_2 = self.nn_computor.kneighbors(d1_cpu, return_distance=True)

        ratio = distances[:, 0]/distances[:, 1]

        pc_nearest = pc2.clone().detach()[:, idx_nn_2[:, 0]]
        return pc_nearest, pc1.new_tensor((ratio < self.matching_ratio).astype(int))


    def nearest_match(self, pc1, pc2, desc1, desc2):
        d1 = desc1.view(desc1.size(0), -1)
        d2 = desc2.view(desc2.size(0), -1)
        if self.normalize_desc:
            d1 = func_nn.normalize(d1, dim=0)
            d2 = func_nn.normalize(d2, dim=0)

        if not self.fitted:
            self.fit(pc2)
            self.fitted = False
        n_neighbors = int(pc1.size(1)*self.nn_ratio)
        idx_nn = self.nn_computor.kneighbors(pc1.detach().t().cpu().numpy(), return_distance=False,
                                             n_neighbors=n_neighbors)
        d_matrix = torch.sum((d1.cpu().unsqueeze(-1) - d2.cpu()[:, idx_nn]) ** 2, 0)
        sorted = np.argsort(d_matrix.cpu().numpy())

        ratio = d_matrix[np.arange(sorted.shape[0]), sorted[:, 0]]/d_matrix[np.arange(sorted.shape[0]), sorted[:, 1]]

        pc_nearest = pc2.clone()[:, idx_nn[np.arange(sorted.shape[0]), sorted[:, 0]]]

        return pc_nearest, ratio < self.matching_ratio


    def bidirectional_matching(self, pc1, pc2, desc1, desc2):
        d1 = desc1.view(desc1.size(0), -1)
        d2 = desc2.view(desc2.size(0), -1)
        if self.normalize_desc:
            d1 = func_nn.normalize(d1, dim=0)
            d2 = func_nn.normalize(d2, dim=0)
        d1_cpu = d1.detach().t().cpu().numpy()

        if not self.fitted:
            self.fit(desc2)
            self.fitted = False

        idx_nn_2 = self.nn_computor.kneighbors(d1_cpu, return_distance=False)[:, 0]

        bi_nn_computor = neighbors.NearestNeighbors(n_neighbors=1, metric='minkowski')
        bi_nn_computor.fit(d1_cpu)
        idx_nn_1 = bi_nn_computor.kneighbors(d2.detach().t().cpu().numpy(), return_distance=False)[:, 0]

        pc_nearest = pc2.clone().detach()[:, idx_nn_2]
        return pc_nearest, pc1.new_tensor((np.arange(idx_nn_2.shape[0]) == idx_nn_1[idx_nn_2]).astype(int))


    def fast_soft_knn(self, pc1, pc2, desc1, desc2):
        d1 = desc1.view(desc1.size(0), -1)
        d2 = desc2.view(desc2.size(0), -1)
        if self.normalize_desc:
            d1 = func_nn.normalize(d1, dim=0)
            d2 = func_nn.normalize(d2, dim=0)
        d1_cpu = d1.detach().t().cpu().numpy()

        if not self.fitted:
            self.fit(desc2)
            self.fitted = False

        idx_nn_2 = self.nn_computor.kneighbors(d1_cpu, return_distance=False)

        d_matrix = torch.sum((d1.unsqueeze(-1) - d2[:, idx_nn_2]) ** 2, 0)
        d_matrix = self.softmax_1d(-self.fact * d_matrix)

        pc_nearest = torch.sum(pc2[:, idx_nn_2] * d_matrix, -1)
        return pc_nearest

    def desc_reweighting(self, pc1, pc2, desc1, desc2):
        p1 = pc1.view(pc1.size(0), -1)
        p2 = pc2.view(pc2.size(0), -1)
        d1 = desc1.view(desc1.size(0), -1)
        d2 = desc2.view(desc2.size(0), -1)
        if self.normalize_desc:
            d1 = func_nn.normalize(d1, dim=0)
            d2 = func_nn.normalize(d2, dim=0)
        p1_cpu = p1.detach().t().cpu().numpy()

        if not self.fitted:
            self.fit(p2)
            self.fitted = False

        idx_nn_2 = self.nn_computor.kneighbors(p1_cpu, return_distance=False)

        d_matrix = torch.sum((d1.unsqueeze(-1) - d2[:, idx_nn_2]) ** 2, 0)
        d_matrix = self.softmax_1d(-self.fact * d_matrix)

        pc_nearest = torch.sum(pc2[:, idx_nn_2] * d_matrix, -1)
        return pc_nearest

    def soft_knn(self, pc1, pc2, *args):
        distance = 1
        if self.use_dst_pt:
            distance *= torch.reciprocal(self.spatial_distance(pc1, pc2).clamp(min=self.eps))
        if self.use_dst_desc:
            if args:
                distance *= self.descriptor_distance(args[0], args[1])
        d_matrix = self.softmax_1d(self.fact * distance)

        pc_nearest = torch.cat([torch.sum(pc2 * prob, 1).unsqueeze(1) for i, prob in enumerate(d_matrix)], 1)

        mean_distance = torch.mean(func_nn.pairwise_distance(pc1.t(), pc_nearest.t(), p=2))

        return pc_nearest

    def soft_outlier_rejection(self, pc1, pc2):
        dist = torch.norm(pc1 - pc2, dim=0)
        mean_dist = torch.mean(dist, 0)
        filter = self.sigmoid((dist - mean_dist * self.reject_ratio - self.eps) * -1e10)

        return filter

    def get_training_layers(self, layers_to_train=None):
        if layers_to_train is None:
            layers_to_train = self.layers_to_train
        if layers_to_train == 'all':
            return []


    def forward(self, pc1, pc2, *args):
        '''
        Return matches from pc2 to pc1

        :param pc1: pc ref
        :param pc2: pc to align
        :return: points of pc2 nn to pc1
        '''

        pc_nearest = pc1.new_zeros(pc1.size())
        indexor = pc1.new_ones(pc1.size(0), pc1.size(2))
        for i, pc in enumerate(pc1):
            if args:
                if self.hard:
                    pc_nearest[i], indexor[i] = self.knn(pc, pc2[i], args[0][i], args[1][i])
                else:
                    pc_nearest[i] = self.knn(pc, pc2[i], args[0][i], args[1][i])
            else:
                if self.hard:
                    pc_nearest[i], indexor[i] = self.knn(pc, pc2[i])
                else:
                    pc_nearest[i] = self.knn(pc, pc2[i])

            if self.outlier_filter:
                indexor[i] = self.soft_outlier_rejection(pc, pc_nearest[i])

        if self.hard or self.outlier_filter:
            return {'nn': pc_nearest, 'inliers': indexor.int()}
        else:
            return {'nn': pc_nearest,}


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
    '''
    print(T)
    net = CPNet(fact=2, outlier_filter=False, use_dst_desc=True, use_dst_pt=True, desc_p=2, pose_solver='svd', knn='hard')
    desc1 = torch.rand(1, 32, nb_pt).to(device)
    desc2 = desc1

    T = net(p1, p2, desc1, desc2)
    print(T['T'])
    '''
    net = MatchNet(use_dst_desc=True, use_dst_pt=True, knn='hard', fact=20000)
    #net_cpu = MatchNet(use_dst_desc=True, use_dst_pt=False, knn='hard_cpu')
    net_cpu = MatchNet(use_dst_desc=True, use_dst_pt=False, knn='fast_soft_knn', fact=1)

    t1 = time.time()
    nearest = net(p1, p2, desc1, desc2)
    t2 = time.time()
    print(nearest['nn'] - T.matmul(p1))
    print('In {}'.format(t2 - t1))
    t2 = time.time()
    nearest = net_cpu(p1, p2, desc1, desc2)
    t3 = time.time()
    print(nearest['nn'] - T.matmul(p1))
    print('In {}'.format(t3 - t2))
