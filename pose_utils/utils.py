import setlog
import torch
import matplotlib.pyplot as plt


logger = setlog.get_logger(__name__)


def mat_proj(T, bVec, homo=False):
    if homo:
        d_size = 1
        for s in bVec.size()[1:]:
            d_size *= s
        homo_bVec = T.new_ones(4, d_size)
        homo_bVec[:3, :] = bVec.view(3, -1)
        tbVec = homo_bVec.view(homo_bVec.size(0), -1).transpose(0, 1).contiguous().unsqueeze(-1)
        tproj = torch.matmul(T, tbVec)
        proj = tproj.transpose(0, 1).contiguous().view(bVec.size())
    else:
        tbVec = bVec.view(bVec.size(0), -1).transpose(0, 1).contiguous().unsqueeze(-1)
        tproj = torch.matmul(T, tbVec)
        proj = tproj.transpose(0, 1).contiguous().view(bVec.size())

    return proj


def depth_map_to_pc(depth_map, K, remove_zeros=False):
    p = [[[i, j, 1] for j in range(depth_map.size(1))] for i in range(depth_map.size(2))]
    p = depth_map.new_tensor(p).transpose(0, 2).contiguous()

    inv_K = K.inverse()
    p_d = (p * depth_map).view(3, -1).transpose(0, 1)

    if remove_zeros:
        p_d = p_d[p_d.nonzero()][:, 0]
    p_d = p_d.transpose(0, 1).contiguous()

    x = mat_proj(inv_K, p_d)

    return x


def toSceneCoord(depth, pose, K, remove_zeros=False):
    x = depth_map_to_pc(depth, K, remove_zeros=remove_zeros)

    X = mat_proj(pose[:3, :], x, homo=True)
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