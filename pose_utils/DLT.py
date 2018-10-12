import setlog
import PIL.Image
import torch
import torchvision
import torchvision.transforms.functional as func
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datasets.custom_quaternion as custom_q
import numpy as np
import random as rd
import torch.autograd as auto


logger = setlog.get_logger(__name__)

def mat_proj(T, bVec, homo=False):
    if homo:
        homo_bVec = torch.ones(4, bVec.size(1), bVec.size(2))
        homo_bVec[:3, :, :] = bVec
        return mat_proj(T, homo_bVec)
    tbVec = bVec.transpose(0, 2).transpose(0, 1).contiguous().unsqueeze(-1)
    tproj = torch.matmul(T, tbVec)
    proj = tproj.transpose(0, 2).transpose(1, 2).contiguous().squeeze()

    return proj

def toSceneCoord(depth, pose, K):

    p = [[[i, j, 1] for j in range(depth.size(1))] for i in range(depth.size(2))]
    p = torch.FloatTensor(p).transpose(0, 2).contiguous()

    inv_K = K.inverse()
    p_d = p * depth
    x = mat_proj(inv_K, p_d)

    X = mat_proj(pose[:3, :], x, homo=True)
    return X

def variable_hook(grad):
    print('variable hook')
    print('grad', grad)

def dlt(hyps, sceneCoord, K, **kwargs):
    grad = kwargs.pop('grad', False)
    cuda = kwargs.pop('cuda', False)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    if len(hyps) < 6:
        raise ArithmeticError('No enought paired points ({}) to compute P with DLT'.format(len(hyps)))

    # Creation of matrix A
    f_iter = True
    for n_hyp, hyp in enumerate(hyps):
        if cuda:
            homo_3Dpt = torch.cat((sceneCoord[:, hyp[1], hyp[0]].squeeze(), torch.Tensor([1]).cuda()), 0)
        else:
            homo_3Dpt = torch.cat((sceneCoord[:, hyp[1], hyp[0]].squeeze(), torch.Tensor([1])), 0)
        if f_iter:
            A = torch.cat((0 * homo_3Dpt, -1 * homo_3Dpt, hyp[1] * homo_3Dpt,) ,0)
            A = torch.stack((A,
                             torch.cat((1 * homo_3Dpt, 0 * homo_3Dpt, -hyp[0] * homo_3Dpt), 0)
                             ), 0)
            f_iter = False
        else:
            A = torch.cat((A,
                          torch.cat((0 * homo_3Dpt, -1 * homo_3Dpt, hyp[1] * homo_3Dpt), 0).unsqueeze(0)
                          ), 0)
            A = torch.cat((A,
                           torch.cat((1 * homo_3Dpt, 0 * homo_3Dpt, -hyp[0] * homo_3Dpt), 0).unsqueeze(0)
                           ), 0)

    (U, S, V) = torch.svd(A)

    p = V[:,-1]


    if p[10].item() < 0: # Diag of rot mat should be > 0
        p = p * -1

    norm = (p[8]**2 + p[9]**2 + p[10]**2)**0.5
    p = p/norm
    P = p.view(3,4)

    if cuda:
        K = K.cuda()

    pose = K.inverse().matmul(P)

    return pose


def plt_pc(pc, ax, pas = 50, color='b'):
    x = pc[0, :, :].view(1, -1).numpy()[0]
    x = [x[i] for i in range(0, len(x), pas)]
    y = pc[1, :, :].view(1, -1).numpy()[0]
    y = [y[i] for i in range(0, len(y), pas)]
    z = pc[2, :, :].view(1, -1).numpy()[0]
    z = [z[i] for i in range(0, len(z), pas)]

    ax.scatter(x, y, z, c=color, depthshade=True)


def draw_hyps(n, width=640, height=480):
    return [[rd.randint(0, width-1), rd.randint(0, height-1)] for _ in range(n)]

if __name__ == '__main__':
    ids = ['frame-000100', 'frame-000125',]#, 'frame-000300' 'frame-000500', 'frame-000400', 'frame-000600', 'frame-000700', 'frame-000800', 'frame-000900']

    pc = list()
    scale = 1/32

    K = torch.zeros(3, 3)
    K[0, 0] = 585
    K[0, 2] = 320
    K[1, 1] = 585
    K[1, 2] = 240

    K *= scale

    K[2, 2] = 1

    for id in ids:
        rgb_im = '/media/nathan/Data/7_Scenes/heads/seq-02/' + id + '.color.png'
        depth_im = '/media/nathan/Data/7_Scenes/heads/seq-02/' + id + '.depth.png'
        pose_im = '/media/nathan/Data/7_Scenes/heads/seq-02/' + id + '.pose.txt'

        im = func.to_tensor(func.resize(PIL.Image.open(rgb_im), int(480*scale))).float()
        depth = func.to_tensor(func.resize(PIL.Image.open(depth_im), int(480*scale), interpolation=0),).float()
        depth[depth==65535] = 0
        depth *= 1e-3

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

        X = toSceneCoord(depth, pose, K)
        hyps = draw_hyps(10, width=640*scale, height=480*scale)

        X_noise = torch.rand(X.size())*1e-4 + X

        dlt_pose = dlt(hyps, X, K)

        print('Real P:')
        print(K.matmul(pose.inverse()[:3, :]))
        print('Real pose:')
        print(pose.inverse()[:3, :])
        print('DLT pose:')
        print(dlt_pose)
        print('Diff:')
        print(dlt_pose - pose.inverse()[:3, :])

        pc.append(X)
        '''
        plt.figure(1)
        grid = torchvision.utils.make_grid(im)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        grid = torchvision.utils.make_grid(depth)
        plt.figure(2)
        plt.imshow(grid.numpy().transpose((1, 2, 0))[:,:,0], cmap=plt.get_cmap('jet'))
        '''

    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    pas = int(250*scale)
    pas = 1

    color = ['c', 'b']

    for n_pc, point_cloud in enumerate(pc):
        plt_pc(point_cloud, ax, pas, color[n_pc])

    plt.show()
