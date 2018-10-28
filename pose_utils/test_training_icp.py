import setlog

logger = setlog.get_logger(__name__)
logger.setLevel('INFO')

import PIL.Image
import torch
import torchvision
import torchvision.transforms.functional as func
import torch.nn.functional as functional
import torch.nn as nn
import datasets.custom_quaternion as custom_q
import pose_utils.ICP as ICP
import pose_utils.utils as utils
import torch.optim as optim
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import networks.CustomArchi as CA


def init_net():
    net = nn.Sequential(
        nn.Conv2d(3, 50, 4),
        nn.ReLU(inplace=True),
        nn.Conv2d(50, 100, 4, stride=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(100, 200, 4, stride=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(200, 100, 4, stride=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(100, 50, 4, stride=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(50, 1, 4),
        nn.Sigmoid()
    )

    return net

def small_init_net():
    net = nn.Sequential(
        nn.Conv2d(3, 50, 4, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(50, 100, 4, stride=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(100, 50, 4, stride=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(50, 1, 2),
        nn.Sigmoid()
    )

    return net


def variable_hook(grad):
    print('variable hook')
    print('grad', grad)


if __name__ == '__main__':
    ids = ['frame-000019','frame-000119', 'frame-000219', 'frame-000319', 'frame-000419', 'frame-000519',
           'frame-000619', 'frame-000719', 'frame-000819', 'frame-000919']
    scale_net = 1 / 2

    scale = 1/4.285714285714286

    K = torch.zeros(3, 3)
    K[0, 0] = 585
    #K[0, 2] = 320
    K[0, 2] = 240
    K[1, 1] = 585
    K[1, 2] = 240

    K *= scale

    K[2, 2] = 1

    root = '/media/nathan/Data/7_Scenes/heads/seq-01/'
    root = '/Users/n.piasco/Documents/Dev/seven_scenes/heads/seq-01/'

    ims = list()
    ims_nn = list()
    depths = list()
    depthsquares = list()
    poses = list()
    pcs = list()
    pcssquare = list()

    for id in ids:
        rgb_im = root + id + '.color.png'
        depth_im = root + id + '.depth.png'
        pose_im = root + id + '.pose.txt'

        ims.append(
            func.normalize(
                func.to_tensor(
                    func.center_crop(
                    func.resize(PIL.Image.open(rgb_im), int(480 * scale)),
                    int(480 * scale))
                ).float(),
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
            )
        )
        ims_nn.append(
            func.to_tensor(
                func.resize(PIL.Image.open(rgb_im), int(480 * scale))
            ).float(),
        )

        depth = func.to_tensor(
                func.center_crop(
                func.resize(PIL.Image.open(depth_im), int(480 * scale), interpolation=0),
                int(480 * scale))
        ).float()
        depth[depth == 65535] = 0
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
        #pose[:3, 3] = pose[:3, 3] * 1e1
        '''
        rot = pose[0:3, 0:3].numpy()
        quat = custom_q.Quaternion(matrix=rot)
        quat._normalise()
        rot = torch.FloatTensor(quat.rotation_matrix)
        pose[:3, :3] = rot
        '''

        poses.append(pose)

        pcs.append(utils.toSceneCoord(depth, pose, K, remove_zeros=True))

    idx = 0
    pc_ref = torch.cat((pcs[0], pcs[1], pcs[2]), 1)
    pc_ref = utils.get_local_map(T=poses[idx], sequences='TestSplit.txt')
    #pc_ref = torch.load('base_model.pth')

    #pc_ref = torch.cat(pcs, 1)

    #torch.save(pc_ref.detach(), 'full_model.pth')

    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')

    pas = 10
    utils.plt_pc(pc_ref, ax, pas, 'b')
    utils.plt_pc(pcs[idx], ax, pas, 'r')

    plt.show()

    pc_ref.requires_grad = False
    #pc_ref = torch.load('base_model.pth')
    pose = poses[1][:3,:]

    im_fwd = ims[1].unsqueeze(0)
    #net = init_net()
    #net = small_init_net()
    net = nn.Sequential(
            CA.PixEncoder(k_size=4, d_fact=4),
            CA.PixDecoder(k_size=4, d_fact=4, out_channel=1, div_fact=2)
    )
    q = utils.rot_to_quat(pose[:3, :3])
    t = pose[:3, 3]

    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-3)
    #net.register_backward_hook(module_hook)
    it = 10000
    n_hyp = 1
    tt_loss = list()
    #nb_pt_total = int(640 * 480 * scale**2)
    nb_pt_total = int(480 * 480 * scale ** 2)
    param_icp = {
        'iter': 3,
        'fact': 2,
        'dnorm': False,
        'outlier': False
    }
    n_pt = int(0.05 * nb_pt_total)

    init_pose = torch.eye(4,4)
    for i in tqdm.tqdm(range(it)):
        optimizer.zero_grad()
        inv_depth_map = net(im_fwd)
        depth_map = 1/inv_depth_map - 1
        Knet = K * scale_net
        Knet[-1,-1] = 1
        print(im_fwd.size())
        print(depth_map.size())
        print(Knet)
        pc_to_align = utils.depth_map_to_pc(depth_map.squeeze(0), Knet)

        loss = 0
        for hyp in range(n_hyp):
            pc_to_align_pruned = pc_to_align.view(3,-1)[:, np.random.choice(int(nb_pt_total * scale_net**2), (n_pt), replace = False)]
            #pc_ref_pruned = pc_ref.view(3,-1)[:, np.random.choice(nb_pt_total * 3, (n_pt)*3, replace = False)]
            pc_ref_pruned = pc_ref
            pose_net, dist = ICP.soft_icp(pc_ref_pruned, pc_to_align_pruned, init_T=init_pose, **param_icp)
            #pose_net, dist = ICP.soft_icp(pc_to_align_pruned, pc_ref_pruned, init_T=init_pose, **param_icp)
            #pose_net = pose_net.inverse()

            t_net = pose_net[:3, 3]

            q_net = utils.rot_to_quat(pose_net[:3, :3])
            #q_loss += torch.mean(functional.pairwise_distance(q_net.view(-1,1), q.view(-1,1)))

            q_loss = torch.mean(functional.pairwise_distance(q_net.view(-1, 1), q.view(-1, 1)))
            t_loss = torch.mean(functional.pairwise_distance(t_net.view(-1, 1), t.view(-1, 1)))

            loss =+ 0.9*torch.max(torch.stack((q_loss, t_loss), 0)) + 0.1*torch.min(torch.stack((q_loss, t_loss), 0))# + 0.05*dist

        #depth_map_loss = torch.mean(functional.pairwise_distance(depth_map.view(-1, 1), depths[1].view(-1, 1), p=1))

        tt_loss.append(loss.item())

        loss.backward()
        optimizer.step()
        if not i%25:
            print('Loss is {} {} (q) {} (t) {} (d) {} (dist)'.format(loss.item(), q_loss.item(), t_loss.item(), 0, dist.item()))
            print('Pose is')
            print(t_net, q_net)
            print('Real pose is')
            print(t, q)

            print(2*torch.acos(torch.dot(q_net, q)) * 180/3.14159260)
            print(torch.sqrt(torch.sum((t_net - t)**2)))
            print(torch.norm(t_net - t))

            fig = plt.figure(3)
            ax = fig.add_subplot(111, projection='3d')
            pas = 1

            utils.plt_pc(pc_ref_pruned, ax, pas, 'b')
            utils.plt_pc(utils.mat_proj(pose_net[:3,:].detach(), pc_to_align_pruned.detach(), homo=True), ax, pas, 'c')

            plt.figure(1)
            grid = torchvision.utils.make_grid(torch.cat(
                (depth_map.detach(),
                 inv_depth_map.detach()),
            ), nrow=2
            )
            plt.imshow(grid.numpy().transpose((1, 2, 0))[:, :, 0], cmap=plt.get_cmap('jet'))
            plt.figure(4)
            grid = torchvision.utils.make_grid(torch.cat(
                (depths[1].unsqueeze(0).detach(),
                 1 / (1 + depths[1]).unsqueeze(0).detach(),),
            ), nrow=2
            )
            plt.imshow(grid.numpy().transpose((1, 2, 0))[:, :, 0], cmap=plt.get_cmap('jet'))
            plt.figure(2)
            grid = torchvision.utils.make_grid(ims_nn[1])
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
            '''
            plt.figure(4)
            d_maps = [ICP.error_map(pc_ref.view(3,-1),
                                    pc_to_align.view(3,-1),
                                    fact, int(640*scale)).unsqueeze(0).detach() for fact in (1, 5, 10, 100)]
            grid = torchvision.utils.make_grid(torch.cat(d_maps, 0), nrow=2)
            plt.imshow(grid.numpy().transpose((1, 2, 0))[:, :, 0], cmap=plt.get_cmap('jet'))
            plt.colorbar()
            '''
            plt.show()

    plt.show()

