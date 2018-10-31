import setlog
import torch
import matplotlib.pyplot as plt
import os
import re
import pathlib as path
import PIL.Image
import torchvision.transforms.functional as func
import PIL.Image


logger = setlog.get_logger(__name__)


def depth_map_to_pc(depth_map, K, remove_zeros=False):
    p = [[[i, j, 1] for j in range(depth_map.size(1))] for i in range(depth_map.size(2))]
    p = depth_map.new_tensor(p).transpose(0, 2)

    inv_K = K.inverse()
    p_d = (p * depth_map).view(3, -1)

    if remove_zeros:
        p_d = p_d[:, depth_map.view(1, -1).squeeze() != 0]

    x = inv_K.matmul(p_d)
    x_homo = x.new_ones(4, x.nelement()//3)
    x_homo[:3, :] = x
    return x_homo


def toSceneCoord(depth, pose, K, remove_zeros=False):
    x = depth_map_to_pc(depth, K, remove_zeros=remove_zeros)

    X = pose.matmul(x)
    return X


def plt_pc(pc, ax, pas = 50, color='b'):
    x = pc[0, :].view(1, -1).cpu().numpy()[0]
    x = [x[i] for i in range(0, len(x), pas)]
    y = pc[1, :].view(1, -1).cpu().numpy()[0]
    y = [y[i] for i in range(0, len(y), pas)]
    z = pc[2, :].view(1, -1).cpu().numpy()[0]
    z = [z[i] for i in range(0, len(z), pas)]

    ax.scatter(x, y, z, c=color, depthshade=True)


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
    K = kwargs.pop('K', [[585, 0.0, 320], [0.0, 585, 240], [0.0, 0.0, 1.0]])
    frame_spacing = kwargs.pop('frame_spacing', 20)
    output_size = kwargs.pop('output_size', 5000)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    # Loading files...
    env_var = os.environ[dataset]
    folders = list()
    with open(env_var + scene + sequences, 'r') as f:
        for line in f:
            fold = 'seq-{:02d}/'.format(int(re.search('(?<=sequence)\d', line).group(0)))
            folders.append(env_var + scene + fold)
    data = list()
    for i, folder in enumerate(folders):
        p = path.Path(folder)
        data += [(i, re.search('(?<=-)\d+', file.name).group(0))
                 for file in p.iterdir()
                 if file.is_file() and '.txt' in file.name]
    poses = list()
    for fold, seq_num in data:
        pose_file = folders[fold] + 'frame-' + seq_num + '.pose.txt'
        pose = T.new_zeros(4, 4)
        with open(pose_file, 'r') as pose_file_pt:
            for i, line in enumerate(pose_file_pt):
                for j, c in enumerate(line.split('\t')):
                    try:
                        pose[i, j] = float(c)
                    except ValueError:
                        pass
        poses.append(pose)

    # Nearest pose search
    eye_mat = T.new_zeros(4, 4)
    eye_mat[0, 0] = eye_mat[1, 1] = eye_mat[2, 2] = eye_mat[3, 3] = 1
    d_poses = [torch.norm(eye_mat - pose.matmul(T.inverse())).item() for pose in poses]
    nearest_idx = sorted(range(len(d_poses)), key=lambda k: d_poses[k])

    # Computing local pc
    K = T.new_tensor(K)
    K[0, :] *= resize_fact
    K[1, :] *= resize_fact
    pcs = list()
    for i in range(num_pc - 1, num_pc*frame_spacing, frame_spacing):
        fold, num = data[nearest_idx[i]]
        file_name = folders[fold] + 'frame-' + num + '.depth.png'
        depth = PIL.Image.open(file_name)
        new_h = int(min(depth.size)*resize_fact)
        if new_h/2 != min(K[0, 2].item(), K[1, 2].item()):
            logger.warn('Resize factor is modifying the 3D geometry!! (fact={})'.format(resize_fact))
        depth = func.to_tensor(
            func.resize(func.resize(depth, new_h*2, interpolation=PIL.Image.NEAREST),
                        new_h, interpolation=PIL.Image.NEAREST)
        ).float()
        depth[depth == 65535] = 0
        depth *= 1e-3
        depth = depth.to(T.device) # move on GPU if necessary

        pcs.append(toSceneCoord(depth, poses[nearest_idx[i]], K, remove_zeros=True))

    # Pruning step
    final_pc = torch.cat(pcs, 1)
    logger.debug('Final points before pruning cloud has {} points'.format(final_pc.size(1)))
    if isinstance(output_size, int):
        indexor = torch.randperm(final_pc.size(1))
        final_pc = final_pc[:, indexor]
        final_pc = final_pc[:, :output_size]
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