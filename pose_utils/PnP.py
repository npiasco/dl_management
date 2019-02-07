import setlog
import PIL.Image
import torch
import torchvision.transforms.functional as func
import pose_utils.utils as utils
import datasets.custom_quaternion as custom_q
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
import networks.ICPNet as ICPNet
import pose_utils.RANSACPose as RSCPose
import pyopengv
import io
import contextlib

logger = setlog.get_logger(__name__)

def reproject_back(pc, K):
    Q = pc.new_tensor([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]])
    keypoints = K.matmul(Q.matmul(pc[0]))
    keypoints = keypoints[:2] / keypoints[2]
    keypoints = torch.round(keypoints)

    return keypoints


def keypoints_to_bearing(keypoints, K, norm=True):
    bearing_vectors = keypoints.new_zeros(3, keypoints.size(1))
    bearing_vectors[:2, :] =  (-keypoints.new_tensor(K[:2, 2]) + keypoints.t()).t()
    #bearing_vectors[:2, :] = keypoints - keypoints.new_tensor(K[:1, 2])
    bearing_vectors[2, :] = K[0, 0]
    if norm:
        bearing_vectors = torch.nn.functional.normalize(bearing_vectors, dim=0)
    return bearing_vectors


def PnP(pc_to_align, pc_ref, desc_to_align, desc_ref, init_T, K, **kwargs):
    verbose = kwargs.pop('verbose', False)
    match_function = kwargs.pop('match_function',  None)
    desc_function = kwargs.pop('desc_function', None)
    fit_pc = kwargs.pop('fit_pc', False)
    pnp_algo = kwargs.pop('pnp_algo', 'epnp')
    '''
        Algo are: KNEIP - GAO - EPNP - TWOPT - GP3P
    '''


    timing = False
    if timing:
        t_beg = time.time()

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    if verbose:
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111, projection='3d')
        plt.ion()
        plt.show()
        pas = 1

    if desc_function is not None:
        desc_ref = desc_function(pc_ref, desc_ref)
    else:
        desc_ref = pc_ref

    if fit_pc:
        match_function.fit(pc_ref[0])
    else:
        match_function.fit(desc_ref[0])

    if timing:
        t = time.time()
    pc_rec = init_T.matmul(pc_to_align)

    if desc_function is not None:
        desc_ta = desc_function(pc_rec, desc_to_align)
    else:
        desc_ta = pc_rec

    res_match = match_function(pc_rec, pc_ref, desc_ta, desc_ref)
    if 'inliers' in res_match.keys():
        pc_to_align = pc_to_align[0, :, res_match['inliers'][0].byte()].unsqueeze(0)
        res_match['nn'] =  res_match['nn'][0, :, res_match['inliers'][0].byte()].unsqueeze(0)

        if pc_to_align.size(2) == 0:
            logger.warning("0 inliers founded, retuturning intial pose")
            return {'T': init_T}

    keypoints = reproject_back(pc_to_align, K.squeeze())

    bearing_vector = keypoints_to_bearing(keypoints, K.squeeze())

    non_nan_idx, _ = torch.min(bearing_vector == bearing_vector, dim=0)
    bearing_vector = bearing_vector[:, non_nan_idx]
    corr3d_pt = res_match['nn'][0, :3, non_nan_idx]

    fio = io.StringIO()

    #with ostream_redirect(stdout=True, stderr=True):

       # help(pow)
    """
    print('Interpected:')
    s = fio.getvalue()
    print(s)
    """
    T = pyopengv.absolute_pose_ransac(bearing_vector.t().cpu().numpy(), corr3d_pt.t().cpu().numpy(),
                                      algo_name=pnp_algo, threshold=0.0002, iterations=1000)
                                      #algo_name = pnp_algo, threshold = 0.0002, iterations = 1000)
    #T = pyopengv.absolute_pose_epnp(bearing_vector.t().cpu().numpy(), corr3d_pt.t().cpu().numpy())
    if pc_to_align.device == 'gpu':
        T = T.cuda()

    if timing:
        print('Iteration on {}s'.format(time.time()-t))

    if verbose:
        # Ploting
        ax1.clear()
        utils.plt_pc(pc_ref[0], ax1, pas, 'b', size=50, marker='*')
        utils.plt_pc(pc_rec[0], ax1, pas, 'r', size=50, marker='o')
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([-1, 1])
        ax1.set_zlim([-1, 1])

        plt.pause(0.1)

    if verbose:
        plt.ioff()
        ax1.clear()
        plt.close()

    match_function.unfit()

    if timing:
        print('ICP converge on {}s'.format(time.time() - t_beg))

    final_T = pc_ref.new_zeros(4, 4)
    final_T[3, 3] = 1.0
    final_T[:3, :] = pc_ref.new_tensor(T)

    return {'T': final_T.unsqueeze(0)}

if __name__ == '__main__':
    ids = ['frame-000100','frame-000150', 'frame-000150']

    scale = 1/16

    K = torch.eye(3, 3)
    K[0, 0] = 585
    K[0, 2] = 320
    K[1, 1] = 585
    K[1, 2] = 240

    K[:2, :] *= scale

    root = '/media/nathan/Data/7_Scenes/heads/seq-02/'
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
    rd_trans[:3, :3] = utils.rotation_matrix(torch.Tensor([1, 0, 0]), torch.Tensor([0.1]))
    #rd_trans[:3, :] = poses[1][:3,:]
    pc_ref = torch.cat((pcs[0], pcs[2]), 1)

    pc_to_align = rd_trans.matmul(pcs[1])
    #pc_to_align = poses[1].inverse().matmul(pcs[1])

    print('Loading finished')

    fig = plt.figure(10)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Before alignement')
    pas = 1

    utils.plt_pc(pc_ref, ax, pas, 'b', size=50)
    utils.plt_pc(pc_to_align, ax, pas, 'r', size=50)

    #T, d = ICPwNet(pc_ref, pc_to_align, torch.eye(4, 4), iter=20, verbose=True,
#                   arg_net={'fact': 2, 'reject_ratio': 1, 'pose_solver': 'svd', })
    match_net_param = {
        'normalize_desc': False,
        'knn': 'fast_soft_knn',
        #'knn': 'hard_cpu',
        #'bidirectional': True,
        'n_neighbors': 1
    }

    #T = PnP(pc_to_align.unsqueeze(0), pc_ref.unsqueeze(0), pc_to_align.unsqueeze(0), pc_ref.unsqueeze(0),
    T=PnP(pc_to_align.unsqueeze(0), pcs[1].unsqueeze(0), pc_to_align.unsqueeze(0), pcs[1].unsqueeze(0),
                torch.eye(4), verbose=True,
                match_function=ICPNet.MatchNet(**match_net_param),
                #pose_function=PoseFromMatching,
                desc_function=None,
                K=K, pnp_algo='epnp')['T'][0]

    pc_aligned = T.inverse().matmul(pc_to_align)
    #pc_aligned = T.matmul(pc_to_align)

    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('After alignement')

    pas = 1

    utils.plt_pc(pc_aligned, ax, pas, 'b', size=50)
    utils.plt_pc(pc_ref, ax, pas, 'c', size=50)


    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('GT')
    pas = 1

    utils.plt_pc(pcs[1], ax, pas, 'b', size=50)
    utils.plt_pc(pc_ref, ax, pas, 'c', size=50)
    print(T.inverse())
    print(poses[1])
    print(torch.matmul(T, poses[1]))

    plt.show()
