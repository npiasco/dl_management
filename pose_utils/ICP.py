import setlog
import PIL.Image
import torch
import torch.nn.functional as nn_func
import torchvision.transforms.functional as func
import pose_utils.utils as utils
import datasets.custom_quaternion as custom_q
import matplotlib.pyplot as plt
import time
import torchvision as tvis
import math
from mpl_toolkits.mplot3d import Axes3D


logger = setlog.get_logger(__name__)


def error_map(pc_ref, pc_to_align, fact, width):
    d_map = torch.zeros((1, pc_to_align.size(-1)//width, width))
    pc_ref_t = pc_ref.transpose(0, 1)
    print('Error map computation...')
    for i, pt in (enumerate(pc_to_align.transpose(0, 1))):
        d_to_pt = torch.sum((pc_ref_t - pt)**2, 1)
        prob = torch.softmax(fact * -d_to_pt, 0)
        p_nearest = torch.sum(pc_ref * prob, 1)
        d_map[0, i//width, i - (i//width)*width] = torch.norm(pt - p_nearest, p=2)
    print('Error map computed!')
    return d_map


def show_outliers(pc_ref, pc_to_align, threshold, width):
    out_map = torch.zeros((1, pc_to_align.size(-1)//width, width))

    print('Outliers map computation...')
    for i, pt in (enumerate(pc_to_align.transpose(0, 1))):
        if torch.norm(pt - pc_ref[:, i], p=2) > threshold:
            out_map[0, i // width, i - (i // width) * width] = 1
    print('Outliers map computed!')
    return out_map


def outlier_filter(pc_nearest, pc_to_align, threshold):
    pc_nearest_filtered = None
    pc_to_align_filtered = None
    for i, pt in enumerate(pc_to_align.transpose(0, 1)):
        if torch.norm(pt - pc_nearest[:, i], p=2) < threshold:
            if pc_nearest_filtered is None:
                pc_nearest_filtered = pc_nearest[:, i].unsqueeze(1)
                pc_to_align_filtered = pc_to_align[:, i].unsqueeze(1)
            else:
                pc_nearest_filtered = torch.cat((pc_nearest_filtered, pc_nearest[:, i].unsqueeze(1)), 1)
                pc_to_align_filtered = torch.cat((pc_to_align_filtered, pc_to_align[:, i].unsqueeze(1)), 1)

    if pc_to_align_filtered is None:
        return outlier_filter(pc_nearest, pc_to_align, 2*threshold)
    else:
        return pc_nearest_filtered, pc_to_align_filtered


def soft_outlier_filter(pc_nearest, pc_to_align, reject_ratio=1):
    dist = torch.norm(pc_nearest - pc_to_align, dim=0)
    mean_dist = torch.mean(dist, 0)
    eps = 1e-5
    filter = nn_func.sigmoid((dist - mean_dist*reject_ratio + eps)*-1e10)

    return filter


def hard_outlier_filter(pc_nearest, pc_to_align, reject_ratio=1):
    filter = soft_outlier_filter(pc_nearest, pc_to_align, reject_ratio=reject_ratio).long()
    pc_nearest = torch.cat([pt.unsqueeze(1) for i, pt in enumerate(pc_nearest.t()) if filter[i]], 1)
    pc_to_align = torch.cat([pt.unsqueeze(1) for i, pt in enumerate(pc_to_align.t()) if filter[i]], 1)
    return pc_nearest, pc_to_align


def weighted_knn(pc_ref, pc_to_align, fact=10):
    pc_ref_t = pc_ref.transpose(0, 1)
    npc_to_align = None
    pc_nearest = None
    mean_distance = 0
    for i, pt in enumerate(pc_to_align.transpose(0, 1)):
        d_to_pt = torch.sum((pc_ref_t - pt)**2, 1)
        d_to_pt = d_to_pt / torch.mean(d_to_pt)
        prob = torch.softmax(fact * -d_to_pt, 0)

        if pc_nearest is None:
            pc_nearest = (pc_ref * prob)
            npc_to_align = pt.unsqueeze(1).repeat(1, prob.size(0)) * prob
        else:
            pc_nearest = torch.cat((pc_nearest, (pc_ref * prob)), 1)
            npc_to_align = torch.cat((npc_to_align, pt.unsqueeze(1).repeat(1, prob.size(0)) * prob), 1)

        mean_distance += torch.norm(pt - torch.sum(pc_ref * prob, 1), p=2)

    return pc_nearest, npc_to_align, mean_distance/(i+1)

def hard_knn(pc_ref, pc_to_align, fact=10, ref_to_targ=False):
    pc_nearest = new_pc_to_align = None
    pc_ref_t = pc_ref.transpose(0, 1)
    dist_matrix = pc_ref.new_zeros(pc_to_align.size(1), pc_ref.size(1))
    mean_distance = 0
    for i, pt in enumerate(pc_to_align.transpose(0, 1)):
        dist_matrix[i, :] = torch.sum((pc_ref_t - pt)**2, 1)

    dist_matrix_all = torch.softmax(fact * -dist_matrix, 0)
    dist_matrix_nearest = torch.softmax(fact * -dist_matrix, 1)

    for i, pt in enumerate(pc_to_align.transpose(0, 1)):
        val, idx_1 = torch.max(dist_matrix_nearest[i, :], 0)
        val, idx_2 = torch.max(dist_matrix_all[:, idx_1], 0)
        if i == idx_2.item():
            #if torch.abs(dist_matrix[i, idx_1] - dist_matrix[idx_2, idx_1]).item() < 1/fact:
            if pc_nearest is None:
                pc_nearest = torch.sum(pc_ref * dist_matrix_nearest[i, :], 1).unsqueeze(1)
                new_pc_to_align = torch.sum(pc_to_align * dist_matrix_all[:, idx_1], 1).unsqueeze(1)
            else:
                pc_nearest = torch.cat((pc_nearest, torch.sum(pc_ref * dist_matrix_nearest[i, :], 1).unsqueeze(1)), 1)
                new_pc_to_align = torch.cat((new_pc_to_align, torch.sum(pc_to_align * dist_matrix_all[:, idx_1], 1).unsqueeze(1)), 1)
            mean_distance += torch.norm(new_pc_to_align[:, -1] - pc_nearest[:, -1], p=2)
    if ref_to_targ:
        for j, pt in enumerate(pc_ref_t):
            val, idx_1 = torch.max(dist_matrix_all[:, j], 0)
            val, idx_2 = torch.max(dist_matrix_nearest[idx_1, :], 0)
            if j == idx_2.item():
                if pc_nearest is None:
                    pc_nearest = torch.sum(pc_ref * dist_matrix_nearest[idx_1, :], 1).unsqueeze(1)
                    new_pc_to_align = torch.sum(pc_to_align * dist_matrix_all[:, j], 1).unsqueeze(1)
                else:
                    pc_nearest = torch.cat((pc_nearest, torch.sum(pc_ref * dist_matrix_nearest[idx_1, :], 1).unsqueeze(1)), 1)
                    new_pc_to_align = torch.cat((new_pc_to_align, torch.sum(pc_to_align * dist_matrix_all[:, j], 1).unsqueeze(1)), 1)
                mean_distance += torch.norm(new_pc_to_align[:, -1] - pc_nearest[:, -1], p=2)

    return new_pc_to_align, pc_nearest, mean_distance/pc_nearest.size(1)
'''

def hard_knn(pc_ref, pc_to_align, fact=10):
    pc_nearest = new_pc_to_align = None
    pc_ref_t = pc_ref.transpose(0, 1)
    dist_matrix = pc_ref.new_zeros(pc_to_align.size(1), pc_ref.size(1))
    mean_distance = 0
    for i, pt in enumerate(pc_to_align.transpose(0, 1)):
        dist_matrix[i, :] = torch.sum((pc_ref_t - pt) ** 2, 1)

    dist_matrix_all = torch.softmax(fact * -dist_matrix, 0)
    dist_matrix_nearest = torch.softmax(fact * -dist_matrix, 1)

    for i, pt in enumerate(pc_to_align.transpose(0, 1)):
        val, idx_1 = torch.max(dist_matrix_nearest[i, :], 0)
        val, idx_2 = torch.max(dist_matrix_all[:, idx_1], 0)
        if i == idx_2.item():
            # if torch.abs(dist_matrix[i, idx_1] - dist_matrix[idx_2, idx_1]).item() < 1/fact:
            if pc_nearest is None:
                pc_nearest = pc_ref[:,idx_1].unsqueeze(1)
                new_pc_to_align = pc_to_align[:, i].unsqueeze(1)
            else:
                pc_nearest = torch.cat((pc_nearest, pc_ref[:,idx_1].unsqueeze(1)), 1)
                new_pc_to_align = torch.cat(
                    (new_pc_to_align, pc_to_align[:, i].unsqueeze(1)), 1)
            mean_distance += torch.norm(new_pc_to_align[:, -1] - pc_nearest[:, -1], p=2)

    return new_pc_to_align, pc_nearest, mean_distance / pc_nearest.size(1)
'''

def soft_knn(pc_ref, pc_to_align, fact=10, d_norm=False):
    logger.debug('Softmax fact is {}'.format(fact))
    pc_nearest = pc_to_align.clone()
    pc_ref_t = pc_ref.transpose(0, 1)
    mean_distance = 0
    for i, pt in enumerate(pc_to_align.transpose(0, 1)):
        d_to_pt = torch.sum((pc_ref_t - pt)**2, 1)
        if d_norm:
            d_to_pt = d_to_pt / torch.mean(d_to_pt)

        prob = nn_func.softmax(fact * 1/d_to_pt, dim=0)
        pc_nearest[:, i] = torch.sum(pc_ref * prob, 1)
        mean_distance += torch.sum((pt - pc_nearest[:, i])**2)

    return pc_nearest, mean_distance/(i+1)

def fast_soft_knn(pc_ref, pc_to_align, fact=10, eps=1e-8):
    logger.debug('Softmax fact is {}'.format(fact))

    pc_ref_extended = pc_ref.new_ones(3*3, pc_ref.size(1))
    pc_to_align_extended = pc_to_align.new_ones(pc_to_align.size(1), 3*3)

    pc_ref_square = pc_ref**2
    pc_ref_extended[0, :] = pc_ref_square[0, :]
    pc_ref_extended[3, :] = pc_ref_square[1, :]
    pc_ref_extended[6, :] = pc_ref_square[2, :]
    pc_ref_extended[1, :] = pc_ref[0, :] * -2
    pc_ref_extended[4, :] = pc_ref[1, :] * -2
    pc_ref_extended[7, :] = pc_ref[2, :] * -2

    pc_to_align_square = pc_to_align**2
    pc_to_align_extended[:, 2] = pc_to_align_square[0, :]
    pc_to_align_extended[:, 5] = pc_to_align_square[1, :]
    pc_to_align_extended[:, 8] = pc_to_align_square[2, :]
    pc_to_align_extended[:, 1] = pc_to_align[0, :]
    pc_to_align_extended[:, 4] = pc_to_align[1, :]
    pc_to_align_extended[:, 7] = pc_to_align[2, :]

    d_matrix = pc_to_align_extended.matmul(pc_ref_extended)
    d_matrix = nn_func.softmax(fact * torch.reciprocal(d_matrix.clamp(min=eps)), dim=1)

    pc_nearest = torch.cat([torch.sum(pc_ref*prob, 1).unsqueeze(1) for i, prob in enumerate(d_matrix)], 1)
    mean_distance = torch.mean(torch.sum((pc_to_align - pc_nearest)**2, 0))
    return pc_nearest, mean_distance


def best_fit_transform(pc_ref, pc_to_align, indexor):
    pc_ref_centroid = torch.sum(pc_ref[:3, :]*indexor, -1)/torch.sum(indexor)
    pc_ref_centred = ((pc_ref[:3, :].t() - pc_ref_centroid).t()*indexor).t()

    pc_to_align_centroid = torch.sum(pc_to_align[:3, :]*indexor, -1)/torch.sum(indexor)
    pc_to_align_centred = ((pc_to_align[:3, :].t() - pc_to_align_centroid).t() * indexor).t()


    H = torch.matmul(pc_to_align_centred.t(), pc_ref_centred)
    logger.debug('SVD on:')
    logger.debug(H)
    U, S, V = torch.svd(H)
    """
    R = torch.matmul(U, V.t())

    # special reflection case
    if torch.det(R) < 0:
       V = (V * -1).t()
       R = torch.matmul(U, V.t())
    """
    if torch.det(U)*torch.det(V) < 0:
        V = V * V.new_tensor([[1, 1, -1], [1, 1, -1], [1, 1, -1]])

    R = torch.matmul(V, U.t())

    # translation
    t = pc_ref_centroid - torch.matmul(R, pc_to_align_centroid)

    # homogeneous transformation
    T = pc_ref.new_zeros(4,4)
    T[:3, :3] = R
    T[:3, 3] = t
    T[3, 3] = 1

    return T


def soft_icp(pc_ref, pc_to_align, init_T, **kwargs):
    iter = kwargs.pop('iter', 100)
    tolerance = kwargs.pop('tolerance', 1e-3)
    unit_fact = kwargs.pop('fact', 1)
    outlier_rejection = kwargs.pop('outlier', False)
    hard_rejection = kwargs.pop('hard_rejection', False)
    distance_norm = kwargs.pop('dnorm', False)
    verbose = kwargs.pop('verbose', False)
    use_hard_nn = kwargs.pop('use_hard_nn', False)
    fixed_fact = kwargs.pop('fixed_fact', False)
    custom_filter = kwargs.pop('custom_filter',  None)
    reject_ratio = kwargs.pop('reject_ratio',  1)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    if verbose:
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111, projection='3d')
        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(111, projection='3d')
        plt.ion()
        plt.show()
        pas = 1

    T = init_T
    # Row data
    row_pc_ref = pc_ref.view(4, -1)
    row_pc_to_align = pc_to_align.view(4, -1)
    indexor = pc_to_align.new_ones(row_pc_to_align.size(-1))

    # First iter
    fact = 1 * unit_fact
    prev_dist = 0

    for i in range(iter):
        logger.debug('Iteration {}'.format(i))
        #t = time.time()
        pc_rec = T.matmul(row_pc_to_align)
        if use_hard_nn:
            pc_rec, pc_nearest, dist = hard_knn(row_pc_ref, pc_rec, fact=fact)
        else:
            #pc_nearest, dist = soft_knn(row_pc_ref, pc_rec, softmax_tool, fact=fact, d_norm=distance_norm)
            pc_nearest, dist = fast_soft_knn(row_pc_ref, pc_rec, fact=fact)
        #print('Elapsed for matching {}'.format(time.time() - t))
        if outlier_rejection:
            indexor = soft_outlier_filter(pc_nearest, pc_rec, reject_ratio)
        if hard_rejection:
            pc_nearest, pc_rec = hard_outlier_filter(pc_nearest, pc_rec, reject_ratio)
            indexor = pc_to_align.new_ones(pc_nearest.size(-1))
        if custom_filter is not None:
            indexor = indexor*custom_filter

        new_T = best_fit_transform(pc_nearest, pc_rec, indexor)
        T = torch.matmul(T, new_T)

        entrop = abs(prev_dist - dist.item())
        fact = unit_fact if fixed_fact else min(1000, max(1, 1/entrop)) * unit_fact

        if entrop < tolerance:
            logger.debug('Done in {} it'.format(i))
            break
        else:
            prev_dist = dist.item()
        #print('Elapsed all {}'.format(time.time() - t))

        if verbose:
            # Ploting
            ax1.clear()
            utils.plt_pc(row_pc_ref, ax1, pas, 'b')
            utils.plt_pc(pc_rec, ax1, pas, 'r')
            ax1.set_xlim([-1, 1])
            ax1.set_ylim([-1, 1])
            ax1.set_zlim([-1, 1])

            ax2.clear()
            utils.plt_pc(pc_nearest, ax2, pas, 'c')
            utils.plt_pc(pc_rec, ax2, pas, 'r')
            ax2.set_xlim([-1, 1])
            ax2.set_ylim([-1, 1])
            ax2.set_zlim([-1, 1])

            plt.pause(0.1)

    if verbose:
        plt.ioff()
        ax1.clear()
        plt.close()
        ax2.clear()
        plt.close()

    pc_rec = T.matmul(row_pc_to_align)
    pc_nearest, dist = fast_soft_knn(row_pc_ref, pc_rec, softmax_tool, fact=1e5) # hard assigment
    if hard_rejection:
        pc_nearest, pc_rec = hard_outlier_filter(pc_nearest, pc_rec, reject_ratio)
        indexor = pc_to_align.new_ones(pc_nearest.size(-1))
    elif outlier_rejection:
        indexor = soft_outlier_filter(pc_nearest, pc_rec, sig_tool, reject_ratio)
    real_error = torch.mean(torch.sum(((pc_rec - pc_nearest)*indexor)**2, 0))
    return T, real_error


if __name__ == '__main__':
    ids = ['frame-000100','frame-000150', 'frame-000150']

    scale = 1/8

    K = torch.zeros(3, 3)
    K[0, 0] = 585
    K[0, 2] = 320
    K[1, 1] = 585
    K[1, 2] = 240

    K *= scale

    K[2, 2] = 1

    root = '/media/nathan/Data/7_Scenes/heads/seq-01/'
    #root = '/Users/n.piasco/Documents/Dev/seven_scenes/heads/seq-01/'

    ims = list()
    depths = list()
    poses = list()
    pcs = list()

    for id in ids:
        rgb_im = root + id + '.color.png'
        depth_im = root + id + '.depth.png'
        pose_im = root + id + '.pose.txt'

        ims.append(func.to_tensor(func.resize(PIL.Image.open(rgb_im), int(480*scale))).float())

        depth = func.to_tensor(func.resize(PIL.Image.open(depth_im), int(480*scale), interpolation=0),).float()
        depth[depth==65535] = 0
        depth *= 1e-3
        depths.append(depth)

        pose = torch.Tensor(4, 4)
        with open(pose_im, 'r') as pose_file_pt:
            for i, line in enumerate(pose_file_pt):
                for j, c in enumerate(line.split('\t')):
                    try:
                        pose[i, j] = float(c)
                    except ValueError:
                        pass

        rot = pose[0:3, 0:3].numpy()
        quat = custom_q.Quaternion(matrix=rot)
        quat._normalise()
        rot = torch.FloatTensor(quat.rotation_matrix)
        pose[:3, :3] = rot

        poses.append(pose)

        pcs.append(utils.toSceneCoord(depth, pose, K, remove_zeros=False))

    rd_trans = torch.eye(4,4)
    #rd_trans[:,3] = torch.FloatTensor([0.5, -1, 1])
    rd_trans[:3, :3] = utils.rotation_matrix(torch.Tensor([1, 0, 0]), torch.Tensor([1]))
    rd_trans[:3, :3] = poses[1][:3,:3]
    pc_ref = torch.cat((pcs[0], pcs[2]), 1)
    pc_ref = pcs[0]
    print(pc_ref.size())

    pc_to_align = rd_trans.matmul(pcs[1])
    pc_to_align = utils.depth_map_to_pc(depths[1], K, remove_zeros=True)

    print('Loading finished')

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    pas = 1

    utils.plt_pc(pc_ref, ax, pas, 'b')
    utils.plt_pc(pc_to_align, ax, pas, 'r')

    print('Before alignement')

    #T, d = soft_icp(pc_to_align, pc_ref, poses[1].inverse(), tolerance=1e-6, iter=100, fact=100, verbose=True, dnorm=False)
    #T, d = soft_icp(pc_ref, pc_to_align, torch.eye(4, 4), tolerance=1e-5, iter=50, fact=2, verbose=True, dnorm=False, use_hard_nn=True, outlier=True)
    T, d = soft_icp(pc_ref, pc_to_align, torch.eye(4, 4), tolerance=1e-6, iter=100, fact=2, verbose=False, dnorm=False,
                    outlier=True, reject_ratio=1)
    pc_aligned = T.matmul(pc_to_align)

    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    pas = 1

    utils.plt_pc(pc_ref, ax, pas, 'b')
    utils.plt_pc(pc_aligned, ax, pas, 'c')

    print('After alignement')

    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    pas = 1

    utils.plt_pc(pc_ref, ax, pas, 'b')
    utils.plt_pc(pcs[1], ax, pas, 'c')
    print('GT')

    print(torch.matmul(T, poses[1].inverse()))
    print(d)
    '''
    plt.figure(4)
    d_maps = [error_map(pc_ref.view(3, -1),
                        pc_aligned.view(3, -1),
                        fact, int(640 * scale)).unsqueeze(0).detach() for fact in (100,)]
    grid = tvis.utils.make_grid(torch.cat(d_maps, 0), nrow=2)
    plt.imshow(grid.numpy().transpose((1, 2, 0))[:, :, 0], cmap=plt.get_cmap('jet'))
    plt.colorbar()
    '''
    plt.show()
