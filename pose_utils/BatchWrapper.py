import pose_utils.ICP as ICP
import torch
import pose_utils.utils as utils
from trainers.minning_function import recc_acces
import numpy as np
import setlog


logger = setlog.get_logger(__name__)


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


def inverse(variable, **kwargs):
    data_to_inv = kwargs.pop('data_to_inv', None)
    offset = kwargs.pop('offset', -1)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    data_to_inv = recc_acces(variable, data_to_inv)

    return 1/data_to_inv + offset


def batched_local_map_getter(variable, **kwargs):
    Ts = kwargs.pop('T', False)
    map_args = kwargs.pop('map_args', dict())

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    Ts = recc_acces(variable, Ts)
    size_pc = map_args.get('output_size', 2000)

    batched_local_maps = Ts.new_zeros(Ts.size(0), 3, size_pc)

    for i, T in enumerate(Ts):
        batched_local_maps[i] = utils.get_local_map(T=T, **map_args)

    return batched_local_maps


def batched_depth_map_to_pc(variable, **kwargs):
    depth_maps = kwargs.pop('depth_maps', None)
    K = kwargs.pop('K', None)
    inverse_depth = kwargs.pop('inverse_depth', True)
    remove_zeros = kwargs.pop('remove_zeros', False)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    batched_depth_maps = recc_acces(variable, depth_maps)
    K = recc_acces(variable, K)

    n_batch, _, height, width = batched_depth_maps.size()
    if remove_zeros:
        if n_batch != 1:
            raise ArithmeticError("Can't stack pc when removing zerors values! (batch_size!=1)")
    else:
        batched_pc = batched_depth_maps.new_zeros(n_batch, 3, height*width)

    for i, depth_maps in enumerate(batched_depth_maps):
        if inverse_depth:
            depth_maps = 1/depth_maps - 1
        if remove_zeros:
            batched_pc = utils.depth_map_to_pc(depth_maps, K[i], remove_zeros).unsqueeze(0)
        else:
            batched_pc[i, :, :] = utils.depth_map_to_pc(depth_maps, K[i], remove_zeros)

    return batched_pc


def batched_pc_pruning(variable, **kwargs):
    pc = kwargs.pop('pc', None)
    pc_desc = kwargs.pop('pc_desc', None)
    mode = kwargs.pop('mode', 'random')
    pruning_fact = kwargs.pop('pruning_fact', 0.1)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    batched_pc = recc_acces(variable, pc)

    n_batch, _, n_pt = batched_pc.size()

    new_n_pt = int(pruning_fact * n_pt)
    step = n_pt//new_n_pt
    new_n_pt = (len(range(0, n_pt, step)))

    batched_pruned_pc = batched_pc.new_zeros(n_batch, 3, new_n_pt)
    if pc_desc is not None:
        batched_pc_desc = recc_acces(variable, pc_desc)
        size_desc = batched_pc_desc.size(1)
        batched_pc_desc = batched_pc_desc.view(n_batch, size_desc, -1)
        batched_pruned_pc_desc =  batched_pc_desc.new_zeros(n_batch, size_desc, new_n_pt)

    for i, pc in enumerate(batched_pc):

        if mode == 'random':
            indexor = np.random.choice(n_pt, new_n_pt, replace = False)
        elif mode == 'regular':
            indexor = range(0, n_pt, step)

        batched_pruned_pc[i, :, :] = pc[:, indexor]
        if pc_desc is not None:
            batched_pruned_pc_desc[i, :, :] = batched_pc_desc[i, :, indexor]

    if pc_desc is not None:
        pruned_data = {'pc': batched_pruned_pc, 'desc': batched_pruned_pc_desc}
    else:
        pruned_data = batched_pruned_pc

    return pruned_data


def batched_outlier_filter(net, variables, **kwargs):
    input_target = kwargs.pop('input_target', list())
    detach_input = kwargs.pop('detach_input', False)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    input = recc_acces(variables, input_target).detach() if detach_input else recc_acces(variables, input_target)

    input = input.view(input.size(0), input.size(1), -1)

    filters = input.new_zeros(input.size(0), input.size(2))

    for i, features_maps in enumerate(input):
        filters[i, :] =  net(features_maps.view(-1, features_maps.size(0))).squeeze()

    return filters


def batched_icp(variable, **kwargs):
    batched_pc_ref = kwargs.pop('pc_ref', None)
    batched_pc_to_align = kwargs.pop('pc_to_align', None)
    batched_ref = kwargs.pop('batched_ref', False)
    batched_init_T = kwargs.pop('init_T', None)
    param_icp = kwargs.pop('param_icp', dict())
    detach_init_pose = kwargs.pop('detach_init_pose', False)
    custom_filter = kwargs.pop('custom_filter', None)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    batched_pc_ref = recc_acces(variable, batched_pc_ref)
    batched_pc_to_align = recc_acces(variable, batched_pc_to_align)
    if batched_init_T is not None:
        batched_init_T = recc_acces(variable, batched_init_T)
        if detach_init_pose:
            batched_init_T = batched_init_T.detach()
            logger.debug('Detatching init pose')
    if custom_filter is not None:
        custom_filter = recc_acces(variable, custom_filter)

    n_batch = batched_pc_to_align.size(0)
    dist = batched_pc_to_align.new_zeros(n_batch, 1)
    poses = {
        'p': batched_pc_to_align.new_zeros(n_batch, 3),
        'q': batched_pc_to_align.new_zeros(n_batch, 4),
        'T': batched_pc_to_align.new_zeros(n_batch, 4, 4),
    }

    for i, pc_to_align in enumerate(batched_pc_to_align):
        if batched_init_T is not None:
            init_T = batched_init_T[i]
        else:
            init_T = pc_to_align.new_zeros(4, 4)
            init_T[0,0] = init_T[1,1] = init_T[2,2] = init_T[3,3] = 1

        current_filter = custom_filter[i, :] if custom_filter is not None else None
        if batched_ref:
            computed_pose, d = ICP.soft_icp(batched_pc_ref[i, :, :], pc_to_align, init_T, **param_icp, custom_filter=current_filter)
        else:
            computed_pose, d = ICP.soft_icp(batched_pc_ref, pc_to_align, init_T, **param_icp, custom_filter=current_filter)
        dist[i] = d
        poses['p'][i, :] = computed_pose[:3, 3]
        poses['q'][i, :] = utils.rot_to_quat(computed_pose[:3, :3])
        poses['T'][i, :, :] = computed_pose

    return {'errors': dist, 'poses': poses}
