import torch
import matplotlib.pyplot as plt
import torchvision as torchvis
import torchvision.transforms.functional as func
import PIL.Image


def image_warp(img, depth, Ks, Kt, T, padding_mode='zeros', mode='bilinear'):
    # img: the source image (where to sample pixels) -- [B, 3, H, W]
    # depth: depth map of the target image -- [B, 1, H, W]
    # K: intrinsec
    # T: pose from source to target (T_{s->t})

    b, _, h, w = depth.size()
    #coord = [[[i/w - 0.5, j/h - 0.5, 1] for j in range(h)] for i in range(w)]
    coord = [[[[i, j , 1] for j in range(h)] for i in range(w)] for _ in range(b)]
    coord = (depth.new_tensor(coord).transpose(1, 3).contiguous()).view(b, 3, -1)
    invK = torch.stack([k.inverse() for k in Kt])

    corr_coord = Ks.matmul(T[:, :3, :3].matmul(depth.view(b, 1, -1) * invK.matmul(coord)) + T[:, :3, 3].unsqueeze(-1)).view(b, 3, h, w)
    offset = depth.new_tensor([w/2, h/2]).unsqueeze(-1).unsqueeze(-1)
    corr_coord = corr_coord[:, :2, :, :] / corr_coord[:, 2, :, :].unsqueeze(1)
    corr_coord =(corr_coord - offset)/offset
    corr_coord = corr_coord.transpose(1, 3).transpose(1, 2)

    projected_img = torch.nn.functional.grid_sample(img, corr_coord, mode=mode, padding_mode=padding_mode)

    return projected_img


if __name__=='__main__':
    ids = ['frame-000115','frame-000110', 'frame-000120']

    scale = 1/2

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

    for id in ids:
        rgb_im = root + id + '.color.png'
        depth_im = root + id + '.depth.png'
        pose_im = root + id + '.pose.txt'

        ims.append(func.to_tensor(func.resize(PIL.Image.open(rgb_im), int(480*scale))).float().unsqueeze(0))

        depth = func.to_tensor(func.resize(PIL.Image.open(depth_im), int(480*scale), interpolation=0),).float()
        depth[depth == 65535] = 0
        depth *= 1e-3
        # depth = 0.0422 * K[0, 0] * depth.reciprocal()
        depths.append(depth.unsqueeze(0))
        pose = torch.Tensor(4, 4)
        with open(pose_im, 'r') as pose_file_pt:
            for i, line in enumerate(pose_file_pt):
                for j, c in enumerate(line.split('\t')):
                    try:
                        pose[i, j] = float(c)
                    except ValueError:
                        pass

        poses.append(pose)
        print(pose)


    print(torch.sum((poses[0][:3, 3] - poses[1][:3, 3])**2)**0.5)
    ccmap = plt.get_cmap('jet', lut=1024)
    fig = plt.figure(1)
    images_batch = torch.cat((depths[0], depths[1],))
    grid = torchvis.utils.make_grid(images_batch, nrow=2)
    plt.imshow(grid.numpy().transpose(1, 2, 0)[:, :, 0], cmap=ccmap)
    plt.colorbar()

    img_wrapped = image_warp(ims[1], depths[0], K.unsqueeze(0),  K.unsqueeze(0), (poses[1].inverse().matmul(poses[0])).unsqueeze(0))
    #img_wrapped = image_warp(ims[1], depths[0].new_ones(depths[0].size()), K, torch.eye(4,4), padding_mode='zeros')
    fig = plt.figure(2)

    images_batch = torch.cat((ims[1], img_wrapped, ims[1], ims[0],))
    grid = torchvis.utils.make_grid(images_batch, nrow=2)
    plt.imshow(grid.numpy().transpose(1, 2, 0))

    fig = plt.figure(3)
    images_batch = torch.abs(torch.mean(ims[0], dim=1) - torch.mean(img_wrapped, dim=1))
    grid = torchvis.utils.make_grid(images_batch, nrow=1)
    plt.imshow(grid.numpy().transpose(1, 2, 0)[:, :, 0], cmap=ccmap)
    plt.colorbar()

    plt.show()

