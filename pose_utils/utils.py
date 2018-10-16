import setlog
import torch
import matplotlib.pyplot as plt


logger = setlog.get_logger(__name__)


def mat_proj(T, bVec, homo=False):
    if homo:
        d_size = 1
        for s in bVec.size()[1:]:
            d_size *= s
        homo_bVec = torch.ones(4, d_size)
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
    p = torch.FloatTensor(p).transpose(0, 2).contiguous()

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
    x = pc[0, :].view(1, -1).numpy()[0]
    x = [x[i] for i in range(0, len(x), pas)]
    y = pc[1, :].view(1, -1).numpy()[0]
    y = [y[i] for i in range(0, len(y), pas)]
    z = pc[2, :].view(1, -1).numpy()[0]
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


def rot_to_quat(m):
    if m[2, 2].item() < 0:
        if m[0, 0].item() > m[1, 1].item():
            t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
            q = torch.stack([m[1, 2] - m[2, 1], t, m[0, 1] + m[1, 0], m[2, 0] + m[0, 2]], 0)
        else:
            t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
            q = torch.stack([m[2, 0] - m[0, 2], m[0, 1] + m[1, 0], t, m[1, 2] + m[2, 1]], 0)
    else:
        if m[0, 0].item() < -m[1, 1].item():
            t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
            q = torch.stack([m[0, 1] - m[1, 0], m[2, 0] + m[0, 2], m[1, 2] + m[2, 1], t], 0)
        else:
            t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
            q = torch.stack([t, m[1, 2] - m[2, 1], m[2, 0] - m[0, 2], m[0, 1] - m[1, 0]], 0)

    q = q * 0.5 / torch.sqrt(t)
    return q
