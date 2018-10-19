import setlog
import PIL.Image
import torch
import torchvision.transforms.functional as func
import pose_utils.utils as utils
import datasets.custom_quaternion as custom_q
import matplotlib.pyplot as plt
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


def soft_knn(pc_ref, pc_to_align, fact=10, d_norm=True):
    pc_nearest = pc_to_align.clone()
    pc_ref_t = pc_ref.transpose(0, 1)
    mean_distance = 0
    for i, pt in enumerate(pc_to_align.transpose(0, 1)):
        d_to_pt = torch.sum((pc_ref_t - pt)**2, 1)
        if d_norm:
            d_to_pt = d_to_pt / torch.mean(d_to_pt)
        prob = torch.softmax(fact * -d_to_pt, 0)
        pc_nearest[:, i] = torch.sum(pc_ref * prob, 1)
        mean_distance += torch.norm(pt - pc_nearest[:, i], p=2)

    return pc_nearest, mean_distance/(i+1)


def best_fit_transform(pc_ref, pc_to_align):
    pc_ref_centroid = torch.mean(pc_ref, -1)
    pc_ref_centred = (pc_ref.transpose(0, 1) - pc_ref_centroid)

    pc_to_align_centroid = torch.mean(pc_to_align, -1)
    pc_to_align_centred = (pc_to_align.transpose(0, 1) - pc_to_align_centroid)

    H = torch.matmul(pc_ref_centred.t(), pc_to_align_centred)

    U, S, V = torch.svd(H)
    R = torch.matmul(U, V.t())

    # special reflection case
    if torch.det(R) < 0:
       V = (V * -1).t()
       R = torch.matmul(U, V.t())

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
    distance_norm = kwargs.pop('dnorm', True)
    verbose = kwargs.pop('verbose', False)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    if verbose:
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111, projection='3d')
        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(111, projection='3d')
        plt.ion()
        plt.show()
        pas = 5

    T = init_T
    # Row data
    row_pc_ref = pc_ref.view(3, -1)
    row_pc_to_align = pc_to_align.view(3, -1)

    # First iter
    fact = 1 * unit_fact
    pc_rec = utils.mat_proj(T[:3, :], row_pc_to_align, homo=True)
    pc_nearest, init_dist = soft_knn(row_pc_ref, pc_rec, fact=fact, d_norm=distance_norm)
    prev_dist = init_dist

    for i in range(iter):
        new_T = best_fit_transform(pc_nearest, pc_rec)
        T = torch.matmul(T, new_T)

        pc_rec = utils.mat_proj(T[:3, :], row_pc_to_align, homo=True)
        pc_nearest, dist = soft_knn(row_pc_ref, pc_rec, fact=fact, d_norm=distance_norm)

        entrop = abs(prev_dist - dist.item())
        fact = min(100, max(1, 1/entrop)) * unit_fact
        if outlier_rejection:
            threshold = math.tanh(fact*0.05- math.exp(1)) + 1.5
            if verbose:
                out_map = show_outliers(pc_nearest, pc_rec, threshold*dist, int(640 * scale)).unsqueeze(0)
                er_map = error_map(row_pc_ref, pc_rec, fact, int(640 * scale)).unsqueeze(0)
                logger.debug('Softmax factor is {}'.format(fact))
                logger.info('Fact {}, thresh {} dist {} init_dist {} entrop {} inv entrop {}'.format(
                    fact, threshold, dist, init_dist, entrop, 1/entrop))

            pc_nearest, pc_rec = outlier_filter(pc_nearest, pc_rec, threshold*dist)

        if entrop < tolerance:
            break
        else:
            prev_dist = dist

        if verbose:
            # Ploting
            #fig1.clf()
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

            """
            plt.figure(4)
            grid = tvis.utils.make_grid(torch.cat((out_map, er_map)))
            print(dist, dist*2)

            plt.imshow(grid.numpy().transpose((1, 2, 0))[:, :, 0], cmap=plt.get_cmap('jet'))
            plt.colorbar()
            """
            plt.pause(0.2)

    logger.debug('Done in {} it'.format(i))
    return T, dist


if __name__ == '__main__':
    ids = ['frame-000100','frame-000150', 'frame-000150']

    scale = 1/16

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

    pc_to_align = utils.mat_proj(rd_trans[:3, :], pcs[1], homo=True)
    pc_to_align = utils.depth_map_to_pc(depths[1], K, remove_zeros=False)

    print('Loading finished')

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    pas = 1

    utils.plt_pc(pc_ref, ax, pas, 'b')
    utils.plt_pc(pc_to_align, ax, pas, 'r')

    print('Before alignement')

    T, d = soft_icp(pc_ref, pc_to_align, torch.eye(4,4), tolerance=1e-6, iter=100, fact=2, verbose=True)
    pc_aligned = utils.mat_proj(T[:3, :], pc_to_align, homo=True)

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
