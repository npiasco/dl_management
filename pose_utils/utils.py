import setlog
import torch
import networks.ResNet as Resnet
import matplotlib.pyplot as plt
import os
import re
import pathlib as path
import PIL.Image
import torchvision.transforms.functional as func
import PIL.Image
import numpy as np
from plyfile import PlyData, PlyElement
import math
import time


logger = setlog.get_logger(__name__)


def model_to_ply(**kwargs):

    file_name = kwargs.pop('file_name', 'model.ply')
    map_args = kwargs.pop('map_args', dict())
    color = kwargs.pop('color', (255, 0, 0))
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    model = get_local_map(**map_args)
    np_model = model[:3, :].t().cpu().detach().numpy()
    np_model = np.array([(p_[0], p_[1], p_[2], color[0], color[1], color[2]) for p_ in np_model], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    el = PlyElement.describe(np_model, 'vertex')
    PlyData([el]).write(file_name)


def gaussian_kernel(kernel_size=15, sigma=3.0, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          (-torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)).float()
                      )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, padding=(kernel_size-1)//2,
                                      kernel_size=kernel_size, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


def projected_depth_map_utils(poor_pc, nn_pc, T, K, **kwargs):

    inliers = kwargs.pop('inliers', None)
    diffuse = kwargs.pop('diffuse', None)
    n_diffuse = kwargs.pop('n_diffuse', 1)
    keep_sources = kwargs.pop('keep_sources', True)
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    Q = poor_pc.new_tensor([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0]])

    size_depth_map = int(math.sqrt(poor_pc.size(1)))
    initial_dmap = poor_pc.new_zeros(1, size_depth_map, size_depth_map)
    proj_poor_pc = K.matmul(Q.matmul(poor_pc))
    proj_poor_pc[:2, :] /= proj_poor_pc[2, :]
    proj_poor_pc[:2, :] = torch.round(proj_poor_pc[:2, :])
    initial_dmap[:, proj_poor_pc[0, :].long(), proj_poor_pc[1, :].long()] = proj_poor_pc[2, :]

    if inliers is not None:
        poor_pc = poor_pc[:, inliers.bytes()]
        nn_pc = nn_pc[:, inliers.bytes()]

    proj_pc = K.matmul(Q.matmul(poor_pc))
    proj_nn = K.matmul(Q.matmul(T.matmul(nn_pc)))

    proj_pc[:2, :] /= proj_pc[2, :]
    proj_pc[:2, :] = torch.round(proj_pc[:2, :])
    proj_nn[:2, :] /= proj_nn[2, :]
    proj_nn[:2, :] = torch.round(proj_nn[:2, :])

    inliers = (torch.min(proj_pc[0, :].int() == proj_nn[0, :].int(),
                         proj_pc[1, :].int() == proj_nn[1, :].int())).squeeze()
    logger.debug('Get {} reprojection inliers'.format(torch.sum(inliers).item()))

    repro_err = poor_pc.new_zeros(1, size_depth_map, size_depth_map)

    if diffuse is not None:
        for i in range(n_diffuse):
            repro_err[:, proj_nn[0, inliers].long(), proj_nn[1, inliers].long()] = \
                (proj_pc[2, inliers] - proj_nn[2, inliers])
            repro_err = diffuse(repro_err.unsqueeze(0)).squeeze(0)
        if keep_sources:
            repro_err[:, proj_nn[0, inliers].long(), proj_nn[1, inliers].long()] = \
                (proj_pc[2, inliers] - proj_nn[2, inliers])
    else:
        repro_err[:, proj_nn[0, inliers].long(), proj_nn[1, inliers].long()] = \
            (proj_pc[2, inliers] - proj_nn[2, inliers])
    # - / +
    final_map = (initial_dmap - repro_err).clamp(min=0)
    return final_map


def depth_map_to_pc(depth_map, K, remove_zeros=False):
    p = [[[i, j, 1] for j in range(depth_map.size(1))] for i in range(depth_map.size(2))]
    p = depth_map.new_tensor(p).transpose(0, 2).contiguous()

    inv_K = K.inverse()
    p_d = (p * depth_map).view(3, -1)

    if remove_zeros:
        indexor = depth_map.view(1, -1).squeeze() != 0
        p_d = p_d[:, indexor ]

    x = inv_K.matmul(p_d)
    x_homo = x.new_ones(4, x.nelement()//3)
    x_homo[:3, :] = x

    if remove_zeros:
        return x_homo, indexor
    else:
        return x_homo


def toSceneCoord(depth, pose, K, remove_zeros=False):
    if remove_zeros:
        x, _ = depth_map_to_pc(depth, K, remove_zeros=remove_zeros)
    else:
        x = depth_map_to_pc(depth, K, remove_zeros=remove_zeros)

    X = pose.matmul(x)
    return X


def plt_pc(pc, ax, pas = 50, color='b', size=10, marker="."):
    '''
    :param marker: https://matplotlib.org/api/markers_api.html
    '''
    x = pc[0, :].view(1, -1).cpu().numpy()[0]
    x = [x[i] for i in range(0, len(x), pas)]
    y = pc[1, :].view(1, -1).cpu().numpy()[0]
    y = [y[i] for i in range(0, len(y), pas)]
    z = pc[2, :].view(1, -1).cpu().numpy()[0]
    z = [z[i] for i in range(0, len(z), pas)]

    ax.scatter(x, y, z, c=color, depthshade=True, s=size, marker=marker)


def rotation_matrix(axis, theta):
    axis = axis/torch.norm(axis)
    a = torch.cos(theta/2.)
    axsin = -axis*torch.sin(theta/2.)
    b = axsin[0]
    c = axsin[1]
    d = axsin[2]

    return torch.FloatTensor([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                              [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                              [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])


def quat_to_rot(q):
    '''

    :param q: [w, x, y, z]
    :return:
    '''
    sq = q**2
    mat = q.new_zeros(3, 3)
    # invs (inverse square length) is only required if quaternion is not already normalised
    invs = 1 / torch.sum(sq)
    mat[0, 0] = ( sq[1] - sq[2] - sq[3] + sq[0])*invs # since sq[0] + sq[1] + sq[2] + sq[3] =1/invs*invs
    mat[1, 1] = (-sq[1] + sq[2] - sq[3] + sq[0])*invs
    mat[2, 2] = (-sq[1] - sq[2] + sq[3] + sq[0])*invs

    tmp1 = q[1]*q[2]
    tmp2 = q[3]*q[0]
    mat[1, 0] = 2.0 * (tmp1 + tmp2)*invs
    mat[0, 1] = 2.0 * (tmp1 - tmp2)*invs

    tmp1 = q[1]*q[3]
    tmp2 = q[2]*q[0]
    mat[2, 0] = 2.0 * (tmp1 - tmp2)*invs
    mat[0, 2] = 2.0 * (tmp1 + tmp2)*invs
    tmp1 = q[2]*q[3]
    tmp2 = q[1]*q[0]
    mat[2, 1] = 2.0 * (tmp1 + tmp2)*invs
    mat[1, 2] = 2.0 * (tmp1 - tmp2)*invs

    return mat


def rot_to_quat(m):
    q = m.new_zeros(4)
    tr = m.trace()
    if tr > 0:
        S = torch.torch.sqrt(tr + 1.0) * 2 # S = 4 * q[0]
        q[0] = 0.25 * S
        q[1] = (m[2, 1] - m[1, 2]) / S
        q[2] = (m[0, 2] - m[2, 0]) / S
        q[3] = (m[1, 0] - m[0, 1]) / S
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        S = torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2 # S = 4 * q[1]
        q[0] = (m[2, 1] - m[1, 2]) / S
        q[1] = 0.25 * S
        q[2] = (m[0, 1] + m[1, 0]) / S
        q[3] = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2 # S = 4 * q[2]
        q[0] = (m[0, 2] - m[2, 0]) / S
        q[1] = (m[0, 1] + m[1, 0]) / S
        q[2] = 0.25 * S
        q[3] = (m[1, 2] + m[2, 1]) / S
    else:
        S = torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2 # S = 4 * q[3]
        q[0] = (m[1, 0] - m[0, 1]) / S
        q[1] = (m[0, 2] + m[2, 0]) / S
        q[2] = (m[1, 2] + m[2, 1]) / S
        q[3] = 0.25 * S
    return q


def get_local_map(**kwargs):
    T = kwargs.pop('T',  None).detach()
    dataset = kwargs.pop('dataset',  'SEVENSCENES')
    scene =  kwargs.pop('scene',  'heads/')
    sequences = kwargs.pop('sequences',  'TrainSplit.txt')
    num_pc = kwargs.pop('num_pc',  8)
    resize_fact = kwargs.pop('resize',  1/16)
    reduce_fact = kwargs.pop('reduce_fact',  2)
    K = kwargs.pop('K', [[585, 0.0, 240], [0.0, 585, 240], [0.0, 0.0, 1.0]])
    frame_spacing = kwargs.pop('frame_spacing', 20)
    output_size = kwargs.pop('output_size', 5000)
    cnn_descriptor = kwargs.pop('cnn_descriptor', False)
    cnn_depth = kwargs.pop('cnn_depth', False)
    cnn_enc = kwargs.pop('cnn_enc', None)
    cnn_dec = kwargs.pop('cnn_dec', None)
    no_grad = kwargs.pop('no_grad', True)
    test_mode = kwargs.pop('test_mode', False)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    timing = False

    if timing:
        tt = time.time()

    # Loading files...
    if hasattr(get_local_map, 'data') is False:
        env_var = os.environ[dataset]
        get_local_map.folders = list()
        with open(env_var + scene + sequences, 'r') as f:
            for line in f:
                fold = 'seq-{:02d}/'.format(int(re.search('(?<=sequence)\d', line).group(0)))
                get_local_map.folders.append(env_var + scene + fold)


        get_local_map.data = list()
        for i, folder in enumerate(get_local_map.folders):
            p = path.Path(folder)
            get_local_map.data += [(i, re.search('(?<=-)\d+', file.name).group(0))
                                   for file in p.iterdir()
                                   if file.is_file() and '.txt' in file.name]

        get_local_map.poses = list()
        for fold, seq_num in get_local_map.data:
            pose_file = get_local_map.folders[fold] + 'frame-' + seq_num + '.pose.txt'
            pose = np.ndarray((4, 4), dtype=np.float32)
            with open(pose_file, 'r') as pose_file_pt:
                for i, line in enumerate(pose_file_pt):
                    for j, c in enumerate(line.split('\t')):
                        try:
                            pose[i, j] = float(c)
                        except ValueError:
                            pass
            get_local_map.poses.append(pose)


    if timing:
        print('Files loading in {}s'.format(time.time() - tt))
        t = time.time()
    # Nearest pose search
    '''
    eye_mat = T.new_zeros(4, 4)
    eye_mat[0, 0] = eye_mat[1, 1] = eye_mat[2, 2] = eye_mat[3, 3] = 1
    d_poses = [torch.norm(eye_mat - pose.matmul(T.inverse())).item() for pose in poses]
    '''
    InvnpT = np.linalg.inv(T.cpu().numpy())
    eye_mat = np.eye(4, 4)
    d_poses = [np.linalg.norm(eye_mat - np.matmul(pose, InvnpT)) for pose in get_local_map.poses]

    nearest_idx = sorted(range(len(d_poses)), key=lambda k: d_poses[k])
    if timing:
        print('NN search in {}s'.format(time.time() - t))
        t = time.time()

    # Computing local pc
    K = T.new_tensor(K)
    K[:2, :] *= resize_fact
    pcs = list()
    if cnn_descriptor:
        descs = list()
    if cnn_descriptor or cnn_depth:
        if hasattr(get_local_map, 'depth') is False:
            get_local_map.depth = dict()
            get_local_map.out_enc = dict()


    for i in range(0, num_pc*frame_spacing, frame_spacing):
        fold, num = get_local_map.data[nearest_idx[i]]
        if cnn_descriptor or cnn_depth:
            if  nearest_idx[i] not in get_local_map.out_enc.keys():
                file_name = get_local_map.folders[fold] + 'frame-' + num + '.color.png'
                im = PIL.Image.open(file_name)
                new_h = int(min(im.size) * resize_fact * reduce_fact) # 2 time depth map by default
                im = func.to_tensor(
                    func.center_crop(
                        func.resize(im, new_h, interpolation=PIL.Image.BILINEAR),
                        new_h
                    )
                ).float()
                im = im.to(T.device) # move on GPU if necessary
                if no_grad:
                    with torch.no_grad():
                        get_local_map.out_enc[nearest_idx[i]] = cnn_enc(im.unsqueeze(0))
                else:
                    get_local_map.out_enc[nearest_idx[i]] = cnn_enc(im.unsqueeze(0))
            if cnn_descriptor:
                desc = get_local_map.out_enc[nearest_idx[i]][cnn_descriptor].squeeze()
                desc = desc.view(desc.size(0), -1)

        if cnn_depth:
            if  nearest_idx[i] not in get_local_map.depth.keys():
                if no_grad:
                    with torch.no_grad():
                        if isinstance(cnn_dec, Resnet.Deconv):
                            get_local_map.depth[nearest_idx[i]] = cnn_dec(get_local_map.out_enc[nearest_idx[i]]['feat'],
                                                                          get_local_map.out_enc[nearest_idx[i]]['res_1'],
                                                                          get_local_map.out_enc[nearest_idx[i]]['res_2']).squeeze(0)
                        else:
                            get_local_map.depth[nearest_idx[i]] = cnn_dec(get_local_map.out_enc[nearest_idx[i]]).squeeze(0)
                else:
                    if isinstance(cnn_dec, Resnet.Deconv):
                        get_local_map.depth[nearest_idx[i]] = cnn_dec(get_local_map.out_enc[nearest_idx[i]]['feat'],
                                                                      get_local_map.out_enc[nearest_idx[i]]['res_1'],
                                                                      get_local_map.out_enc[nearest_idx[i]]['res_2']).squeeze(0)
                    else:
                        get_local_map.depth[nearest_idx[i]] = cnn_dec(get_local_map.out_enc[nearest_idx[i]]).squeeze(0)

            depth = torch.reciprocal(get_local_map.depth[nearest_idx[i]].clamp(min=1e-8)) - 1  # Need to inverse the depth
            pcs.append(
                toSceneCoord(depth, torch.from_numpy(get_local_map.poses[nearest_idx[i]]).to(T.device),
                             K, remove_zeros=False))
            if cnn_descriptor:
                descs.append(desc)
        else:
            file_name = get_local_map.folders[fold] + 'frame-' + num + '.depth.png'
            depth = PIL.Image.open(file_name)
            new_h = int(min(depth.size)*resize_fact)
            if new_h/2 != min(K[0, 2].item(), K[1, 2].item()):
                logger.warn('Resize factor is modifying the 3D geometry!! (fact={})'.format(resize_fact))
            depth = func.to_tensor(
                func.center_crop(
                    func.resize(func.resize(depth, new_h*2, interpolation=PIL.Image.NEAREST),
                                new_h, interpolation=PIL.Image.NEAREST),
                    new_h
                )
            ).float()
            depth[depth == 65535] = 0
            depth *= 1e-3
            depth = depth.to(T.device) # move on GPU if necessary
            if cnn_descriptor:
                desc = desc[:, depth.view(1, -1).squeeze() != 0]
                descs.append(desc)
            pcs.append(toSceneCoord(depth, torch.from_numpy(get_local_map.poses[nearest_idx[i]]).to(T.device),
                                    K, remove_zeros=True))

    if timing:
        print('PC creation in {}s'.format(time.time() - t))
        t = time.time()

    # Pruning step
    final_pc = torch.cat(pcs, 1)
    if cnn_descriptor:
        cnn_desc_out = torch.cat(descs, 1)
    logger.debug('Final points before pruning cloud has {} points'.format(final_pc.size(1)))
    if not isinstance(output_size, bool):
        indexor = torch.randperm(final_pc.size(1))
        final_pc = final_pc[:, indexor]
        final_pc = final_pc[:, :output_size]
        if cnn_descriptor:
            cnn_desc_out = cnn_desc_out[:, indexor]
            cnn_desc_out = cnn_desc_out[:, :output_size]

    if timing:
        print('Pruning in {}s'.format(time.time() - t))

    if not test_mode:
        del get_local_map.depth
        del get_local_map.out_enc

    if cnn_descriptor:
        return final_pc, cnn_desc_out
    else:
        return final_pc


if __name__ == '__main__':
    axe = torch.tensor([0.33, 0.33, 0.33])
    angle = torch.tensor([3.14159260 / 8])
    rot_mat = rotation_matrix(axe, angle)
    print(rot_mat)
    quat = rot_to_quat(rot_mat)
    print(quat)
    rot = quat_to_rot(quat)
    print(rot)
    q = rot_to_quat(rot)
    print(q)
    print(rot.matmul(rot_mat.t()))
    print(rot_mat.matmul(rot.t()))
    print(2*torch.acos(torch.abs(torch.dot(q, quat))) * 180/3.14156092)
    print(rot_to_quat(rot))
    print(rot_to_quat(rot.t()))
    print(rot_to_quat(torch.eye(3,3)))