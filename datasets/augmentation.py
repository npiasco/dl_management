import pose_utils.utils as utils
import datasets.multmodtf as tf
import setlog
import matplotlib.pyplot as plt
import os
import torch.utils.data as data
import torch
import random
import torchvision as tv


logger = setlog.get_logger(__name__)


def creat_new_sample(sample, zoom=0.2, reduce_fact=2, tilte_angle=1, final_size_depth_map=56):
    Q = torch.tensor([[1.0, 0, 0, 0],
                      [0, 1.0, 0, 0],
                      [0, 0, 1.0, 0]])

    K = torch.from_numpy(sample['K'])
    new_K = K.clone().detach()
    new_K[:2, :] /= reduce_fact

    D = torch.max(sample['depth'].view(-1))
    fov = 2*torch.atan2(K[0, 2], 2*K[0, 0])
    z_offest = random.random()*D*zoom
    max_angle = torch.atan2(D*torch.tan(fov/2), D - z_offest) - fov/2
    theta_x = random.choice([1, -1]) * (random.random() * 0.5 + 0.5) * max_angle

    theta_y = random.choice([1, -1]) * (random.random() * 0.5 + 0.5) * max_angle
    theta_z = random.choice([1, -1]) * random.random() * tilte_angle

    new_pose = torch.eye(4,4)
    new_pose[2, 3] = -z_offest
    new_pose[:3, :3] = utils.rotation_matrix(torch.tensor([1.0, 0, 0]), torch.tensor([theta_x])).matmul(
        new_pose[:3, :3])
    new_pose[:3, :3] = utils.rotation_matrix(torch.tensor([0, 1.0, 0]), torch.tensor([theta_y])).matmul(
        new_pose[:3, :3])
    new_pose[:3, :3] = utils.rotation_matrix(torch.tensor([0, 0, 1.0]), torch.tensor([theta_z])).matmul(
        new_pose[:3, :3])

    _, w, h = sample['rgb'].size()
    w = round(w/reduce_fact)
    h = round(h/reduce_fact)

    ori_pc, _ = utils.depth_map_to_pc(sample['depth'], K, remove_zeros=True)
    move_pc = new_pose.matmul(ori_pc)
    new_depth_maps = torch.zeros(1, w, h)
    repro = new_K.matmul(Q.matmul(move_pc))
    coord = (repro[:2] / repro[2]).round().long()
    coord[0, :] = coord[0, :].clamp(min=0, max=h-1)
    coord[1, :] = coord[1, :].clamp(min=0, max=w-1)
    #flat_coord = coord[0, :]*h + coord[1, :]
    #u_flat_coord = torch.unique(flat_coord)
    new_depth_maps[:, coord[1, :], coord[0, :]] = repro[2, :]

    gap_filed_depth = remove_gap(sample['depth'], sample['depth'].view(-1)==0)
    ori_pc = utils.depth_map_to_pc(gap_filed_depth, K, remove_zeros=False)
    move_pc = new_pose.matmul(ori_pc)
    repro = new_K.matmul(Q.matmul(move_pc))
    coord = (repro[:2] / repro[2]).round().long()
    coord[0, :] = coord[0, :].clamp(min=0, max=h - 1)
    coord[1, :] = coord[1, :].clamp(min=0, max=w - 1)
    new_image = torch.zeros(3, w, h)
    colors = sample['rgb'].view(3, -1)
    new_image[:, coord[1, :], coord[0, :]] = colors
    new_image = remove_gap(new_image,
                           ((new_image[0, :, :] + new_image[1, :, :] + new_image[2, :, :]) == 0).view(-1))

    '''
    for fidx in u_flat_coord:
        indexor = fidx == flat_coord
        if torch.sum(indexor) > 1:
            idx_min = torch.argmin(repro[2, indexor])
            idx_h = flat_coord[idx_min]//h
            idx_w = flat_coord[idx_min] - idx_h*h
            new_depth_maps[:, idx_w, idx_h] = repro[2, idx_min]
            new_image[:, idx_w, idx_h] = colors[:, idx_min]
    '''
    combined_new_pose = new_pose.matmul(torch.from_numpy(sample['pose']['T']))
    full_new_pose = {
        'T': combined_new_pose.numpy(),
        'position': new_pose[:3, 3].numpy(),
        'orientation': utils.rot_to_quat(new_pose[:3, :3]).numpy(),
    }

    return {'depth': torch.nn.functional.interpolate(new_depth_maps.unsqueeze(0),
                                                     size=final_size_depth_map, mode='nearest').squeeze(0),
            'rgb': new_image,
            'K':new_K.numpy(),
            'pose': full_new_pose}


def remove_gap(depth_map, zero_ids):
    ori_size = depth_map.size()
    width = ori_size[-1]
    c = ori_size[0]
    depth_map = depth_map.view(c, -1)
    lenght = depth_map.size(-1)
    clear_depth_map = depth_map.clone().detach()

    for id, is_zero in enumerate(zero_ids):
        if is_zero:
            search_idx = 0
            is_curr_zero = zero_ids[id+search_idx]

            incrementation = True
            while is_curr_zero:
                if search_idx >= 0:
                    if incrementation:
                        search_idx += 1
                else:
                    if incrementation:
                        search_idx -= 1

                incrementation = not incrementation
                search_idx *= -1

                if id%width + search_idx < 0 or id%width + search_idx >= width or id + search_idx > lenght - 1:
                    search_idx *= -1
                    if id%width + search_idx < 0 or id%width + search_idx >= width or id + search_idx > lenght - 1:
                        break

                is_curr_zero = zero_ids[id + search_idx]
            id = min(id, min(depth_map.size(1) - 1 - search_idx, clear_depth_map.size(1) - 1))
            clear_depth_map[:, id] = depth_map[:, id + search_idx]
    depth_map = depth_map.view(ori_size)
    return clear_depth_map.view(ori_size)


def save_aug_dataset(dataset, folder, n_forwards=10):
    try:
        os.mkdir(folder)
    except:
        print('{} exist'.format(folder))
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    for n_forward in range(n_forwards):
        try:
            os.mkdir(folder + 'seq-0{}'.format(n_forward))
        except:
            print('{} exist'.format(folder + 'seq-0{}'.format(n_forward)))
        for i, b in enumerate(dataloader):
            file_base_name = 'seq-0{}/frame-{}'.format(n_forward, i)
            im = tv.transforms.functional.to_pil_image(b['rgb'].squeeze(0))
            im.save(folder + file_base_name + ".color.png", "PNG")
            depth = tv.transforms.functional.to_pil_image((b['depth'].squeeze(0)*1e3).int(), mode='I')
            depth.save(folder + file_base_name + ".depth.png", "PNG", bytes=8)
            with open(folder + file_base_name + '.pose.txt', 'w') as f:
                for l in b['pose']['T'].squeeze(0).numpy():
                    for num in l:
                        f.write("%16.7e\t" % num)
                    f.write('\n')


if __name__ == '__main__':
    import datasets.SevenScene as SevenS
    aug_tf = {
        'first': (tf.CenterCrop(480),),
        'rgb': (tf.ToTensor(), ),
        'depth': (tf.ToTensor(), tf.DepthTransform())
    }

    std_tf = {
        'first': (tf.Resize(256),  tf.RandomCrop(224),),
        'rgb': (tf.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05),
                tf.ToTensor(),
                tf.Normalize(mean=[0.4684, 0.4624, 0.4690], std=[0.2680, 0.2659, 0.2549])),
        'depth': (tf.Resize(56), tf.ToTensor(), tf.DepthTransform())
    }
    root = os.environ['SEVENSCENES'] + 'heads/'

    train_aug_dataset = SevenS.AugmentedTrain(root=root,
                                              transform=aug_tf,
                                              final_depth_size=256,
                                              reduce_fact=1.85,
                                              zoom_percentage=0.15)

    save_aug_dataset(train_aug_dataset, os.environ['SEVENSCENES'] + 'aug_heads/')

    """
    train_dataset = SevenS.Train(root=root,
                                 transform=std_tf, )

    m_dataset = SevenS.MultiDataset(type='train', root=os.environ['SEVENSCENES'],
                                    folders=['heads/', ]*1, transform=std_tf,
                                    general_options={'used_mod': ('rgb', 'depth')})

    print(len(m_dataset))
    dataloader = data.DataLoader(train_aug_dataset, batch_size=2, shuffle=True, num_workers=0)


    fig = plt.figure(1)
    fig2 = plt.figure(2)
    for i, b in enumerate(dataloader):
        fig.clear()
        plt.figure(1)
        SevenS.show_batch(b)
        fig2.clear()
        plt.figure(2)
        SevenS.show_batch_mono(b)
        plt.show()
        del b


    """