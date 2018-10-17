import pose_utils.ICP as ICP
import torch
import pose_utils.utils as utils
from trainers.minning_function import recc_acces
import numpy as np


def add_variable(variable, **kwargs):
    value = kwargs.pop('value', None)
    load = kwargs.pop('load', False)
    source = kwargs.pop('source', None)

    if load:
        value = torch.load(value)
    source = recc_acces(variable, source)
    new_var = source.new_tensor(value)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    return new_var


def batched_depth_map_to_pc(variable, **kwargs):
    depth_maps = kwargs.pop('depth_maps', None)
    K = kwargs.pop('K', None)
    remove_zeros = kwargs.pop('remove_zeros', False)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    batched_depth_maps = recc_acces(variable, depth_maps)
    K = recc_acces(variable, K)

    n_batch, _, height, width = batched_depth_maps.size()
    batched_pc = batched_depth_maps.new_zeros(n_batch, 3, height*width)

    for i, depth_maps in enumerate(batched_depth_maps):
        batched_pc[i, :, :] = utils.depth_map_to_pc(depth_maps, K, remove_zeros)

    return batched_pc


def batched_pc_pruning(variable, **kwargs):
    pc = kwargs.pop('pc', None)
    mode = kwargs.pop('mode', 'random')
    pruning_fact = kwargs.pop('pruning_fact', 0.1)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    batched_pc = recc_acces(variable, pc)
    n_batch, _, n_pt = batched_pc.size()

    new_n_pt = int(pruning_fact * n_pt)
    batched_pruned_pc = batched_pc.new_zeros(n_batch, 3, new_n_pt)

    for i, pc in enumerate(batched_pc):
        if mode == 'random':
            batched_pruned_pc[i, :, :] = pc[:, np.random.choice(n_pt, new_n_pt, replace = False)]

    return batched_pruned_pc


def batched_icp(variable, **kwargs):
    batched_pc_ref = kwargs.pop('pc_ref', None)
    batched_pc_to_align = kwargs.pop('pc_to_align', None)
    batched_ref = kwargs.pop('batched_ref', False)
    batched_init_T = kwargs.pop('init_T', None)
    param_icp = kwargs.pop('param_icp', dict())

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    batched_pc_ref = recc_acces(variable, batched_pc_ref)
    batched_pc_to_align = recc_acces(variable, batched_pc_to_align)
    if batched_init_T is not None:
        batched_init_T = recc_acces(variable, batched_init_T)

    n_batch = batched_pc_to_align.size(0)
    poses = {
        'p': batched_pc_to_align.new_zeros(n_batch, 3),
        'q': batched_pc_to_align.new_zeros(n_batch, 4),
    }

    for i, pc_to_align in enumerate(batched_pc_to_align):
        if batched_init_T is not None:
            init_T = batched_init_T[i, :]
        else:
            init_T = pc_to_align.new_zeros(4, 4)
            init_T[0,0] = init_T[1,0] = init_T[2,2] = init_T[3,3] = 1

        if batched_ref:
            computed_pose, _ = ICP.soft_icp(batched_pc_ref[i, :, :], pc_to_align, init_T, **param_icp)
        else:
            computed_pose, _ = ICP.soft_icp(batched_pc_ref, pc_to_align, init_T, **param_icp)

        poses['p'][i, :] = computed_pose[3, :3]
        poses['q'][i, :] = utils.rot_to_quat(computed_pose[:3, :3])

    return poses
