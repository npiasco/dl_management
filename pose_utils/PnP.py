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
import pyopengv
import numpy as np


logger = setlog.get_logger(__name__)


def reproject_back(pc, K):
    Q = pc.new_tensor([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]])
    keypoints = K.matmul(Q.matmul(pc[0]))
    keypoints = keypoints[:2] / keypoints[2]
    #keypoints = torch.round(keypoints)

    return keypoints


def keypoints_to_bearing(keypoints, K, norm=True):
    bearing_vectors = keypoints.new_zeros(3, keypoints.size(1))
    bearing_vectors[:2, :] =  (-K[:2, 2].clone().detach() + keypoints.t()).t()
    #bearing_vectors[:2, :] = keypoints - keypoints.new_tensor(K[:1, 2])
    bearing_vectors[2, :] = K[0, 0]
    if norm:
        bearing_vectors = torch.nn.functional.normalize(bearing_vectors, dim=0)
    return bearing_vectors


def PnP(pc_to_align, pc_ref, desc_to_align, desc_ref, init_T, K, **kwargs):
    match_function = kwargs.pop('match_function',  None)
    desc_function = kwargs.pop('desc_function', None)
    fit_pc = kwargs.pop('fit_pc', False)
    pnp_algo = kwargs.pop('pnp_algo', 'GAO')
    inliers_threshold = kwargs.pop('inliers_threshold', 0.1)
    ransac_threshold = kwargs.pop('ransac_threshold', 0.0002)
    iterations = kwargs.pop('iterations', 1000)
    '''
        Algo are: KNEIP - GAO - EPNP - TWOPT - GP3P
    '''


    timing = False
    if timing:
        t_beg = time.time()

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

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

        if pc_to_align.size(2) < 4:
            logger.warning("Less than 4 inliers founded, retuturning intial pose")
            return {'T': init_T}

    keypoints = reproject_back(pc_to_align, K.squeeze())

    bearing_vector = keypoints_to_bearing(keypoints, K.squeeze())

    non_nan_idx, _ = torch.min(bearing_vector == bearing_vector, dim=0)
    bearing_vector = bearing_vector[:, non_nan_idx]
    corr3d_pt = res_match['nn'][0, :3, non_nan_idx]

    T = pyopengv.absolute_pose_ransac(bearing_vector.t().cpu().numpy(), corr3d_pt.t().cpu().numpy(),
                                  algo_name=pnp_algo, threshold=ransac_threshold, iterations=iterations)

    with open("ransac_inliers.txt", 'r') as f:
        inliers = int(f.read())
                                      #algo_name = pnp_algo, threshold = 0.0002, iterations = 1000)
    #T = pyopengv.absolute_pose_epnp(bearing_vector.t().cpu().numpy(), corr3d_pt.t().cpu().numpy())
    if pc_to_align.device == 'gpu':
        T = T.cuda()

    if timing:
        print('Iteration on {}s'.format(time.time()-t))

    match_function.unfit()

    if timing:
        print('Pnp converge on {}s'.format(time.time() - t_beg))

    inliers_ratio = inliers / pc_to_align.size(2)

    final_T = pc_ref.new_zeros(4, 4)
    final_T[3, 3] = 1.0
    final_T[:3, :] = pc_ref.new_tensor(T)

    if inliers_ratio < inliers_threshold:
        T_diff = torch.norm(init_T[0] - final_T)
        logger.debug('Not enought inliers (ratio: {})'.format(inliers_ratio))
        logger.debug('Diff in pose is {})'.format(T_diff.item()))
        return {'T': init_T}

    return {'T': final_T.unsqueeze(0)}


def intersec(x_r, u_dir, weights=None):
    '''
    Compute the position of the point that minimise the distance between line defined by 3D points x_r and vector u_dir
    '''

    n_ex = len(x_r)
    A = np.zeros((3*n_ex, 3))
    B = np.zeros((3*n_ex))

    for i, x in enumerate(x_r):
        u = u_dir[i]
        A[i * 3, 1] = u[2]
        A[i * 3, 2] = -u[1]
        A[i * 3 + 1, 0] = -u[2]
        A[i * 3 + 1, 2] = u[0]
        A[i * 3 + 2, 0] = u[1]
        A[i * 3 + 2, 1] = -u[0]
        B[i * 3] = x[1]*u[2] - x[2]*u[1]
        B[i * 3 + 1] = x[2]*u[0] - x[0]*u[2]
        B[i * 3 + 2] = x[0]*u[1] - x[1]*u[0]

        if weights:
            A[i * 3:i * 3 + 3, :] *= np.sqrt(weights[i])
            B[i * 3:i * 3 + 3] *= np.sqrt(weights[i])

    x = np.linalg.lstsq(A, B, rcond=None)

    return x[0]

def average_rotation(Rs, weights=None):
    '''
    https://stackoverflow.com/questions/12374087/average-of-multiple-quaternions
    '''
    if weights:
        qs = [utils.rot_to_quat(torch.from_numpy(R)).numpy().reshape(4, 1)*weights[i] for i, R in enumerate(Rs)]
    else:
        qs = [utils.rot_to_quat(torch.from_numpy(R)).numpy().reshape(4, 1) for R in Rs]

    Q = np.concatenate(qs, axis=1)
    M = np.matmul(Q, np.transpose(Q))
    w, v = np.linalg.eig(M)

    max_eig = np.argmax(w)
    return utils.quat_to_rot(torch.from_numpy(v[:, max_eig].real)).numpy()


def rPnP(pc_to_align, pc_refs, desc_to_align, desc_refs, inits_T, K, **kwargs):
    match_function = kwargs.pop('match_function',  None)
    desc_function = kwargs.pop('desc_function', None)
    fit_pc = kwargs.pop('fit_pc', False)
    pnp_algo = kwargs.pop('pnp_algo', 'NISTER')
    ransac_threshold = kwargs.pop('ransac_threshold', 0.0002)
    iterations = kwargs.pop('iterations', 1000)
    '''
        Algo are: STEWENIUS - NISTER - SEVENPT - EIGHTPT
    '''

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    Rr = list()
    u_dir = list()
    x_r = list()
    inliers = list()

    for n_pc, pc_ref in enumerate(pc_refs):
        desc_ref = desc_refs[n_pc]
        init_T = inits_T[n_pc]

        if desc_function is not None:
            desc_ref = desc_function(pc_ref, desc_ref)
        else:
            desc_ref = pc_ref

        if fit_pc:
            match_function.fit(pc_ref[0])
        else:
            match_function.fit(desc_ref[0])

        pc_rec = inits_T[0].matmul(pc_to_align)

        if desc_function is not None:
            desc_ta = desc_function(pc_rec, desc_to_align)
        else:
            desc_ta = pc_rec

        filtered_pc_to_align = pc_to_align.clone().detach()
        res_match = match_function(pc_rec, pc_ref, desc_ta, desc_ref)
        if 'inliers' in res_match.keys():
            filtered_pc_to_align = pc_to_align[0, :, res_match['inliers'][0].byte()].unsqueeze(0)
            res_match['nn'] =  res_match['nn'][0, :, res_match['inliers'][0].byte()].unsqueeze(0)

            if filtered_pc_to_align.size(2) < 6:
                logger.warning("Less than 6 inliers founded, retuturning intial pose")
                return {'T': init_T}

        keypoints_1 = reproject_back(filtered_pc_to_align , K.squeeze())
        bearing_vector_1 = keypoints_to_bearing(keypoints_1, K.squeeze())
        nn_match = init_T.inverse().matmul(res_match['nn'])
        keypoints_2 = reproject_back(nn_match, K.squeeze())
        bearing_vector_2 = keypoints_to_bearing(keypoints_2, K.squeeze())

        non_nan_idx, _ = torch.min(bearing_vector_1 == bearing_vector_1, dim=0)
        bearing_vector_1 = bearing_vector_1[:, non_nan_idx]
        bearing_vector_2 = bearing_vector_2[:, non_nan_idx]

        Tr = pyopengv.relative_pose_ransac(bearing_vector_1.t().cpu().numpy(), bearing_vector_2.t().cpu().numpy(),
                                           algo_name=pnp_algo, threshold=ransac_threshold, iterations=iterations)
        match_function.unfit()

        with open("ransac_inliers.txt", 'r') as f:
            inlier = int(f.read())
        inliers.append(inlier)
        Tr_w = np.eye(4)
        Tr_w[:3, :] = Tr
        Tr_w = np.matmul(np.linalg.inv(Tr_w), init_T[0].cpu().numpy())
        Rr.append(Tr_w[:3, :3])
        x_r.append(init_T[0, :3, 3].cpu().numpy())
        u_dir.append((np.matmul(init_T[0, :3, :3].cpu().numpy(), -Tr[:3, 3])) / np.linalg.norm(
                    np.matmul(init_T[0, :3, :3].cpu().numpy(), -Tr[:3, 3]))
               )


    inliers_norm = inliers
    '''
    med_inlier = np.median(inliers)
    x_r = [x for i, x in enumerate(x_r) if inliers[i] > med_inlier]
    u_dir = [u for i, u in enumerate(u_dir) if inliers[i] > med_inlier]
    Rr = [R for i, R in enumerate(Rr) if inliers[i] > med_inlier]
    inliers_norm = [inlier for i, inlier in enumerate(inliers) if inlier > med_inlier]
    '''
    t = intersec(x_r, u_dir, weights=inliers_norm)
    T = np.eye(4)

    T[:3, :3] = average_rotation(
        Rr,
        weights=inliers_norm
    )
    T[:3, 3] = t
    T_torch = pc_to_align.new_tensor(T)
    return {'T': T_torch.unsqueeze(0)}


def PnPfrom2D(pc_to_align, pc_refs, desc_to_align, desc_refs, inits_T, K, **kwargs):
    match_function = kwargs.pop('match_function',  None)
    desc_function = kwargs.pop('desc_function', None)
    pnp_param = kwargs.pop('pnp_param',
                           {'pnp_algo': 'GAO',
                            'inliers_threshold': 0.1,
                            'ransac_threshold': 0.0002,
                            'iterations': 1000})
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    Rr = list()
    u_dir = list()
    x_r = list()
    inliers = list()
    pc_match_function = ICPNet.MatchNet(normalize_desc=False,
                                        knn='bidirectional',
                                        ).to(pc_to_align.device)
    # First 3D points triangulation
    """
    if desc_function is not None:
        desc_ref_0 = desc_function(pc_refs[0], desc_refs[0])
        desc_ref_1 = desc_function(pc_refs[1], desc_refs[1])
    else:
        desc_ref_0 = pc_refs[0]
        desc_ref_1 = pc_refs[1]
    """
    #res_match = pc_match_function(pc_refs[0], pc_refs[1], desc_ref_0, desc_ref_1)
    res_match = pc_match_function(pc_refs[0], pc_refs[1], pc_refs[0], pc_refs[1])
    pc_match_function.unfit()

    if 'inliers' in res_match.keys():
        filtered_pc_ref_0 = pc_refs[0][0, :, res_match['inliers'][0].byte()].unsqueeze(0)
        filtered_desc_ref_0 = desc_refs[0][0, :, res_match['inliers'][0].byte()].unsqueeze(0)
        filtered_pc_ref_1 = res_match['nn'][0, :, res_match['inliers'][0].byte()].unsqueeze(0)
    else:
        filtered_pc_ref_0 = pc_refs[0]
        filtered_desc_ref_0 = desc_refs[0]
        filtered_pc_ref_1 = res_match['nn']

    mean_pc = (filtered_pc_ref_0 + filtered_pc_ref_1)/2

    pc_ref_0 = inits_T[0].inverse().matmul(mean_pc)
    keypoints_0 = reproject_back(pc_ref_0, K.squeeze())
    bearing_vector_0 = keypoints_to_bearing(keypoints_0, K.squeeze())

    #nn_match_1 = inits_T[1].inverse().matmul(filtered_pc_ref_1)
    nn_match_1 = inits_T[1].inverse().matmul(mean_pc)
    keypoints_1 = reproject_back(nn_match_1, K.squeeze())
    bearing_vector_1 = keypoints_to_bearing(keypoints_1, K.squeeze())

    R_r = inits_T[0][0, :3, :3].t().matmul(inits_T[1][0, :3, :3])
    t_r = inits_T[0][0, :3, :3].t().matmul(inits_T[1][0, :3, 3] - inits_T[0][0, :3, 3])

    tri_points = pyopengv.triangulation_triangulate2(bearing_vector_0.t().cpu().numpy(),
                                                    bearing_vector_1.t().cpu().numpy(),
                                                    t_r.cpu().numpy(),
                                                    R_r.cpu().numpy())
    tri_pc_ref = filtered_pc_ref_0.clone().detach()
    tri_pc_ref[:, :3, :] = (inits_T[0][0, :3, :3].matmul(torch.from_numpy(tri_points).t().float().to(tri_pc_ref.device)) \
                            + inits_T[0][0, :3, 3].view(3, 1)).unsqueeze(0)

    return PnP(pc_to_align, tri_pc_ref, desc_to_align, filtered_desc_ref_0, inits_T[0], K,
               **pnp_param, match_function=match_function, desc_function=desc_function)

if __name__ == '__main__':
    ids = ['frame-000090','frame-000099', 'frame-000110', 'frame-000140']

    scale = 1/16

    K = torch.eye(3, 3)
    K[0, 0] = 585
    K[0, 2] = 320
    K[1, 1] = 585
    K[1, 2] = 240

    K[:2, :] *= scale

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

    pc_to_align = poses[1].inverse().matmul(pcs[1]).unsqueeze(0)
    desc_to_align = poses[1].inverse().matmul(pcs[1]).unsqueeze(0)
    pc_ref = [pcs[0].unsqueeze(0), pcs[2].unsqueeze(0)]#, pcs[3].unsqueeze(0)]
    desc_ref = [pcs[0].unsqueeze(0), pcs[2].unsqueeze(0)]#, pcs[3].unsqueeze(0)]
    inits_T = [poses[0].unsqueeze(0), poses[2].unsqueeze(0), poses[1].unsqueeze(0)]

    fig = plt.figure(10)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Before alignement')
    pas = 1
    utils.plt_pc(pc_to_align.squeeze(0), ax, pas, 'b', size=50)
    utils.plt_pc(torch.cat((pcs[0], pcs[2])), ax, pas, 'r', size=50)

    match_net_param = {
        'normalize_desc': False,
        'knn': 'fast_soft_knn',
        #'knn': 'hard_cpu',
        #'bidirectional': True,
        'n_neighbors': 1
    }

    #T=PnP(pc_to_align.unsqueeze(0), pcs[1].unsqueeze(0), pc_to_align.unsqueeze(0), pcs[1].unsqueeze(0),
    '''
    T=rPnP(pc_to_align, pc_ref, desc_to_align, desc_ref, inits_T,
           match_function=ICPNet.MatchNet(**match_net_param),
           desc_function=None, iterations=1000, K=K, ransac_threshold=1e-7)['T']
    '''
    T = PnPfrom2D(pc_to_align, pc_ref, desc_to_align, desc_ref, inits_T,
                  match_function=ICPNet.MatchNet(**match_net_param),
                  desc_function=None, K=K)['T']

    pc_aligned = T.matmul(pc_to_align)
    #pc_aligned = T.matmul(pc_to_align)

    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('After alignement')

    pas = 1

    utils.plt_pc(pc_aligned.squeeze(0), ax, pas, 'b', size=50)
    utils.plt_pc(torch.cat((pcs[0], pcs[2])), ax, pas, 'c', size=50)


    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('GT')
    pas = 1

    utils.plt_pc(pcs[1], ax, pas, 'b', size=50)
    utils.plt_pc(torch.cat((pcs[0], pcs[2])), ax, pas, 'c', size=50)
    print(T)
    print(poses[1])
    print(torch.matmul(T.inverse(), poses[1]))

    plt.show()
