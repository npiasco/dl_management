import setlog
import PIL.Image
import torch
import torchvision.transforms.functional as func
import pose_utils.utils as utils
import datasets.custom_quaternion as custom_q
import matplotlib.pyplot as plt
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


def outlier_filter(pc_nearest, pc_to_align, mean_distance):
    for i, pt in enumerate(pc_to_align.transpose(0, 1)):
        if mean_distance < 1 and torch.norm(pt - pc_nearest[:, i], p=2) > mean_distance**0.5:
            pc_nearest[:, i] *= 0
            pc_to_align[:, i] *= 0

    return pc_nearest, pc_to_align


def soft_knn(pc_ref, pc_to_align, fact=10):
    pc_nearest = pc_to_align.clone()
    pc_ref_t = pc_ref.transpose(0, 1)
    mean_distance = 0
    for i, pt in enumerate(pc_to_align.transpose(0, 1)):
        d_to_pt = torch.sum((pc_ref_t - pt)**2, 1)
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
    T = torch.eye(4,4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def soft_icp(pc_ref, pc_to_align, init_T, **kwargs):
    iter = kwargs.pop('iter', 100)
    tolerance = kwargs.pop('tolerance', 1e-3)
    unit_fact = kwargs.pop('fact', 1)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    T = init_T

    # Row data
    row_pc_ref = pc_ref.view(3, -1)
    row_pc_to_align = pc_to_align.view(3, -1)


    # First iter
    fact = 1 * unit_fact
    pc_rec = utils.mat_proj(T[:3, :], row_pc_to_align, homo=True)
    pc_nearest, dist = soft_knn(row_pc_ref, pc_rec, fact=fact)
    prev_dist = dist
    for i in range(iter):
        new_T = best_fit_transform(pc_nearest, pc_rec)

        T = torch.matmul(T, new_T)

        pc_rec = utils.mat_proj(T[:3, :], row_pc_to_align, homo=True)
        pc_nearest, dist = soft_knn(row_pc_ref, pc_rec, fact=fact)
        pc_nearest, pc_rec = outlier_filter(pc_nearest, pc_rec, dist)

        entrop = abs(prev_dist - dist.item())
        fact = max(1/entrop, 1) * unit_fact
        # fact = unit_fact ** max(1 / entrop, 1)

        logger.debug('Softmax factor is {}'.format(fact))
        if entrop < tolerance:
            break
        else:
            prev_dist = dist

    logger.debug('Done in {} it'.format(i))

    return T, dist


if __name__ == '__main__':
    ids = ['frame-000100','frame-000125', 'frame-000150']

    scale = 1/32

    K = torch.zeros(3, 3)
    K[0, 0] = 585
    K[0, 2] = 320
    K[1, 1] = 585
    K[1, 2] = 240

    K *= scale

    K[2, 2] = 1

    root = '/media/nathan/Data/7_Scenes/heads/seq-02/'
    root = '/Users/n.piasco/Documents/Dev/seven_scenes/heads/seq-01/'

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
    print(pc_ref.size())

    pc_to_align = utils.mat_proj(rd_trans[:3, :], pcs[1], homo=True)
    pc_to_align = utils.depth_map_to_pc(depths[1], K, remove_zeros=True)

    print('Loading finished')

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    pas = 1

    utils.plt_pc(pc_ref, ax, pas, 'b')
    utils.plt_pc(pc_to_align, ax, pas, 'r')

    print('Before alignement')

    T, d = soft_icp(pc_ref, pc_to_align, torch.eye(4,4), tolerance=1e-3, iter=100, fact=2)
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

    plt.show()
