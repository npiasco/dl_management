import setlog
import PIL.Image
import torch
import torchvision.transforms.functional as func
import torch.nn.functional as functional
import datasets.custom_quaternion as custom_q
import pose_utils.DLT as utils
import matplotlib.pyplot as plt


logger = setlog.get_logger(__name__)

def soft_knn(pc_ref, pc_to_align):

    return pc_ref

def best_fit_transform(pc_ref, pc_to_align):
    row_pc_ref = pc_ref.view(3, -1)
    pc_ref_centroid = torch.mean(row_pc_ref, -1)
    pc_ref_centred = (row_pc_ref.transpose(0, 1) - pc_ref_centroid)

    row_pc_to_align = pc_to_align.view(3, -1)
    pc_to_align_centroid = torch.mean(row_pc_to_align, -1)
    pc_to_align_centred = (row_pc_to_align.transpose(0, 1) - pc_to_align_centroid)

    H = torch.matmul(pc_ref_centred.t(), pc_to_align_centred)
    U, S, V = torch.svd(H)
    R = torch.matmul(U, V.t())

    # special reflection case
    if torch.det(R) < 0:
       V.t()[:3,:] = V[:3,:] * -1
       R = torch.matmul(U, V.t())

    # translation
    print(pc_ref_centroid)
    print(pc_to_align_centroid)
    print(torch.matmul(R, pc_to_align_centroid))
    t = pc_ref_centroid - torch.matmul(R, pc_to_align_centroid)

    # homogeneous transformation
    T = torch.eye(4,4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T

def soft_icp(pc_ref, pc_to_align, init_T, **kwargs):
    iter = kwargs.pop('iter', 10)
    tolerance = kwargs.pop('tolerance', 1e-3)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    T = init_T

    for i in range(iter):
        pc_rec = utils.mat_proj(T[:3, :], pc_to_align, homo=True)

        pc_nearest = soft_knn(pc_ref, pc_rec)
        new_T = best_fit_transform(pc_nearest, pc_rec)

        T = torch.matmul(T, new_T)
        print('Iter {}'.format(i))
        print(T)


    return T

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

if __name__ == '__main__':
    ids = ['frame-000100','frame-000125']

    scale = 1/32

    K = torch.zeros(3, 3)
    K[0, 0] = 585
    K[0, 2] = 320
    K[1, 1] = 585
    K[1, 2] = 240

    K *= scale

    K[2, 2] = 1

    #root = '/media/nathan/Data/7_Scenes/heads/seq-02/'
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

        pcs.append(utils.toSceneCoord(depth, pose, K))

    rd_trans = torch.eye(3,4)
    rd_trans[:,3] = torch.FloatTensor([0.5, -1, 1])
    #rd_trans[:3, :3] = rotation_matrix(torch.Tensor([1, 0, 0]), torch.Tensor([1]))
    rd_trans[:3, :3] = poses[1][:3,:3]
    pc_ref = pcs[0]

    pc_to_align = utils.mat_proj(rd_trans, pcs[1], homo=True)

    print(pc_to_align.size())

    print('Loading finished')

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    pas = 1

    utils.plt_pc(pc_ref, ax, pas, 'b')
    utils.plt_pc(pc_to_align, ax, pas, 'r')

    print('Before alignement')

    T = soft_icp(pc_ref, pc_to_align, torch.eye(4,4))[:3,:]
    pc_aligned = utils.mat_proj(T[:3, :], pc_to_align, homo=True)

    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    pas = 1

    utils.plt_pc(pc_ref, ax, pas, 'b')
    utils.plt_pc(pc_aligned, ax, pas, 'c')


    print('After alignement')
    plt.show()