import pose_utils.ICP as ICP
import pose_utils.PnP as PnP
import torch
import torch.nn.functional as nn_func
import pose_utils.utils as utils
from trainers.minning_function import recc_acces
import numpy as np
import setlog
import pose_utils.RANSACPose as RSCPose
import trainers.bilinear_sampler as bsm
import time as tm

logger = setlog.get_logger(__name__)


def bilinear_wrapping(variables, **kwargs):
    img_source = kwargs.pop('img_source', None)
    depth_map = kwargs.pop('depth_map', None)
    Ks = kwargs.pop('Ks', None)
    Kt = kwargs.pop('Kt', None)
    T_s = kwargs.pop('T_s', None)
    T_t = kwargs.pop('T_t', None)
    multiple_proj = kwargs.pop('multiple_proj', False)
    resize_K = kwargs.pop('resize_K', False)
    param_sampler = kwargs.pop('param_sampler', dict())

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    timing = False

    if timing:
        t_init = tm.time()

    target_depth_map = recc_acces(variables, depth_map)
    T_t = recc_acces(variables, T_t)

    if multiple_proj:
        listed_var = variables[multiple_proj]
        n_im = len(listed_var) - 1
        Kt = recc_acces(variables, Kt).clone()
        Ks = [recc_acces(listed_var[i + 1], Ks).clone() for i in range(0, n_im)]
        Ks = torch.cat(Ks, dim=0)
        T_s = [recc_acces(listed_var[i + 1], T_s) for i in range(0, n_im)]
        T_s = torch.cat(T_s, dim=0)
        img_sources = [recc_acces(listed_var[i + 1], img_source) for i in range(0, n_im)]
        img_sources = torch.cat(img_sources , dim=0)
        n_batch, _, h, w = target_depth_map.size()

        T = torch.matmul(torch.inverse(T_s), T_t.repeat(n_im, 1, 1))

        if h != img_sources.size(2) or w != img_sources.size(3):
            if resize_K:
                ratio = h / img_sources.size(2)
                Kt[:, :2, :] *= ratio
                Ks[:, :2, :] *= ratio
            img_sources_resized = nn_func.interpolate(img_sources, size=(h, w), mode='bilinear', align_corners=True)
            wrapped_im = bsm.image_warp(img_sources_resized,
                                        target_depth_map.repeat(n_im, 1, 1, 1),
                                        Ks,
                                        Kt.repeat(n_im, 1, 1), T, **param_sampler)
        else:
            wrapped_im = bsm.image_warp(img_sources,
                                        target_depth_map.repeat(n_im, 1, 1, 1),
                                        Ks,
                                        Kt.repeat(n_im, 1, 1), T, **param_sampler)
        wrapped_im = torch.split(wrapped_im, n_batch, dim=0)
    else:
        img_source = recc_acces(variables, img_source)
        Ks = recc_acces(variables, Ks).clone()
        Kt = recc_acces(variables, Kt).clone()

        T_s = recc_acces(variables, T_s)

        #T = torch.stack([t_s.inverse().matmul(T_t[i]) for i, t_s in enumerate(T_s)])
        T = torch.matmul(torch.inverse(T_s), T_t)
        _, _, h, w = target_depth_map.size()
        if h != img_source.size(2) or w != img_source.size(3):
            if resize_K:
                ratio = h / img_source.size(2)
                Kt[:, :2, :] *= ratio
                Ks[:, :2, :] *= ratio
            img_source_resized = nn_func.interpolate(img_source, size=(h, w), mode='bilinear', align_corners=True)
            wrapped_im = bsm.image_warp(img_source_resized, target_depth_map , Ks, Kt, T, **param_sampler)
        else:
            wrapped_im = bsm.image_warp(img_source, target_depth_map, Ks, Kt, T, **param_sampler)

    if timing:
        t = tm.time()
        print('Elapsed {} (total)'.format(t - t_init))

    return wrapped_im


def pnp(nets, variable, **kwargs):
    pc_to_align = kwargs.pop('pc_to_align', None)
    pc_ref = kwargs.pop('pc_ref', None)
    desc_to_align = kwargs.pop('desc_to_align', None)
    desc_ref = kwargs.pop('desc_ref', None)
    K = kwargs.pop('K', None)
    init_T = kwargs.pop('init_T', None)
    inv_init_T = kwargs.pop('inv_init_T', None)
    relative_pnp = kwargs.pop('relative_pnp', False)
    only_pc_for_triangulation = kwargs.pop('only_pc_for_triangulation', False)
    param_pnp = kwargs.pop('param_pnp', dict())

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    pc_to_align = recc_acces(variable, pc_to_align)
    pc_ref = recc_acces(variable, pc_ref)
    desc_to_align = recc_acces(variable, desc_to_align)
    desc_ref = recc_acces(variable, desc_ref)
    init_T = recc_acces(variable, init_T)
    K = recc_acces(variable, K)

    if relative_pnp:
        init_T = [T['T'] for T in init_T]
        pc_ref = [init_T[i].matmul(pc) for i, pc in enumerate(pc_ref)]
        PnP_out = PnP.PnPfrom2D(pc_to_align, pc_ref, desc_to_align, desc_ref, init_T, K,
                                desc_function=(nets[0] if len(nets) > 1 else None),
                                match_function=(nets[1] if len(nets) > 1 else nets[0]),
                                only_pc_for_triangulation=only_pc_for_triangulation,
                                pnp_param=param_pnp)
    else:
        if isinstance(pc_ref, (list, tuple)):
            poses, inliers = list(), list()
            for i in range(len(pc_ref)):
                if init_T[i].size(0) != 1:
                    raise NotImplementedError('No implementation of batched PnP')
                if inv_init_T:
                    init_T[i] = init_T[i][0, :].inverse().unsqueeze(0)

                result = PnP.PnP(pc_to_align, pc_ref[i], desc_to_align, desc_ref[i], init_T[i], K,
                                 desc_function=(nets[0] if len(nets) > 1 else None),
                                 match_function=(nets[1] if len(nets) > 1 else nets[0]),
                                 return_inliers_ratio=True, unfit=True, **param_pnp)
                poses.append(result['T'])
                inliers.append(result['inliers'])
                """
                if inliers[-1] > param_pnp.get('inliers_threshold', 0.1):
                    break
                """
            # (nets[1] if len(nets) > 1 else nets[0]).unfit() TODO: inverse the maching sens
            PnP_out = {'T': poses[np.argmax(inliers)]}
        else:
            if init_T.size(0) != 1:
                raise NotImplementedError('No implementation of batched PnP')
            if inv_init_T:
                init_T = init_T[0, :].inverse().unsqueeze(0)

            PnP_out = PnP.PnP(pc_to_align, pc_ref, desc_to_align, desc_ref, init_T, K,
                              desc_function=(nets[0] if len(nets)>1 else None),
                              match_function=(nets[1] if len(nets)>1 else nets[0]),
                              **param_pnp)

    if inv_init_T:
        PnP_out['T'] = PnP_out['T'][0, :].inverse().unsqueeze(0)

    return {'T': PnP_out['T'],
            'q': utils.rot_to_quat(PnP_out['T'][0,:3,:3]).unsqueeze(0),
            'p': PnP_out['T'][0, :3, 3].unsqueeze(0)
            }


def global_point_net_forward(pointnet, variables, **kwargs):
    pc1 = kwargs.pop('pc1', None)
    pc2 = kwargs.pop('pc2', None)
    desc1 = kwargs.pop('desc1', None)
    desc2 = kwargs.pop('desc2', None)
    detach_inputs = kwargs.pop('detach_inputs', True)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    pc1 = recc_acces(variables, pc1)
    pc2 = recc_acces(variables, pc2)
    desc1 = recc_acces(variables, desc1)
    desc2 = recc_acces(variables, desc2)
    if detach_inputs:
        pc1, pc2, desc1, desc2 = pc1.detach(), pc2.detach(), desc1.detach(), desc2.detach()

    s_pc1 = pc1.size(2)
    s_pc2 = pc2.size(2)

    out = pointnet(torch.cat((pc1, pc2), 2), torch.cat((desc1, desc2), 2))

    pdec1, pdec2 = torch.split(out, (s_pc1, s_pc2), 2)

    return {'desc1': pdec1, 'desc2': pdec2}


def corrected_depth_map_getter(variable, **kwargs):
    poor_pc = kwargs.pop('poor_pc', None)
    nn_pc = kwargs.pop('nn_pc', None)
    T  = kwargs.pop('T', None)
    K  = kwargs.pop('K', None)
    inliers = kwargs.pop('inliers', None)
    diffuse = kwargs.pop('diffuse', None)
    filter_param = kwargs.pop('filter_param', dict())
    filter_loop_param = kwargs.pop('filter_loop_param', dict())

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    poor_pc = recc_acces(variable, poor_pc)
    nn_pc = recc_acces(variable, nn_pc)
    T = recc_acces(variable, T)
    K = recc_acces(variable, K)

    if inliers is not None:
        inliers = recc_acces(variable, inliers)

    if diffuse is not None:
        diffuse = utils.gaussian_kernel(**filter_param).to(poor_pc.device)

    b, _, n_pc = poor_pc.size()
    size_depth_map = int(n_pc**0.5)

    final_maps = poor_pc.new_zeros(b, 1, size_depth_map, size_depth_map)

    for i, pc in enumerate(poor_pc)\
            :
        if inliers is not None:
            inlier = inliers[i]
        else:
            inlier = None

        final_maps[i] = utils.projected_depth_map_utils(pc, nn_pc[i], T[i], K[i],
                                                        diffuse=diffuse,
                                                        inliers=inlier,
                                                        **filter_loop_param)
    return final_maps


def add_variable(variable, **kwargs):
    value = kwargs.pop('value', None)
    load = kwargs.pop('load', False)
    source = kwargs.pop('source', None)
    repeat = kwargs.pop('repeat', False)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    if load:
        value = torch.load(value)
    source = recc_acces(variable, source)
    new_var = source.new_tensor(value)
    if repeat:
        new_var = new_var.unsqueeze(0)
        new_var_size = new_var.size()
        new_size = [1 for _ in range(len(new_var_size))]
        new_size[0] = repeat
        new_var = new_var.repeat(new_size)

    return new_var


def inverse(variable, **kwargs):
    data_to_inv_name = kwargs.pop('data_to_inv', None)
    offset = kwargs.pop('offset', -1)
    eps = kwargs.pop('eps', 1e-8)
    fact = kwargs.pop('fact', 1)
    bounded = kwargs.pop('bounded', False)
    max_depth = kwargs.pop('max_depth', False)
    multiples_instance = kwargs.pop('multiples_instance', False)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    data_to_inv = recc_acces(variable, data_to_inv_name)

    if multiples_instance:
        if isinstance(data_to_inv[0], list):
            data_to_inv = [data[-1] for data in data_to_inv]
        n_batch, _, _, _ = data_to_inv[0].size()
        data_to_inv = torch.cat(data_to_inv, dim=0)

    if max_depth:
        delta = (max_depth**2 + 4 * fact * max_depth)**0.5
        b = (-max_depth + delta) / (2 * fact)
        a = 1/b - 1
        inv_data = (torch.reciprocal(data_to_inv + a) - b)*fact
    elif not bounded:
        inv_data = torch.reciprocal(data_to_inv.clamp(min=eps)*fact) + offset
    else:
        inv_data = torch.reciprocal(data_to_inv*fact + offset)

    if multiples_instance:
        inv_data = torch.split(inv_data, n_batch, dim=0)

    return inv_data


def add_random_transform(variable, **kwargs):
    ori_T = kwargs.pop('original_T', False)
    noise_factor = kwargs.pop('noise_factor', 1e-1)

    if kwargs:

        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    ori_T = recc_acces(variable, ori_T)
    noise_T = ori_T.new_zeros(ori_T.size())

    for i, T in enumerate(ori_T):
        noise_R = utils.rotation_matrix(torch.rand(3), torch.rand(1)*noise_factor)
        noise_t = torch.rand(3)*noise_factor
        noise = ori_T.new_zeros(4, 4)
        noise[:3, :3] = noise_R
        noise[:3, 3] = noise_t
        noise[3, 3] = 1
        noise_T[i] = noise.matmul(T)

    return noise_T


def fast_icp(nets, variable, **kwargs):
    pc_to_align = kwargs.pop('pc_to_align', None)
    pc_ref = kwargs.pop('pc_ref', None)
    desc_to_align = kwargs.pop('desc_to_align', None)
    desc_ref = kwargs.pop('desc_ref', None)
    init_T = kwargs.pop('init_T', None)
    inv_init_T = kwargs.pop('inv_init_T', None)
    param_icp = kwargs.pop('param_icp', dict())
    filter_inliers = kwargs.pop('filter_inliers', False)
    filter_score = kwargs.pop('filter_score', False)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    pc_to_align = recc_acces(variable, pc_to_align)
    pc_ref = recc_acces(variable, pc_ref)
    desc_to_align = recc_acces(variable, desc_to_align)
    desc_ref = recc_acces(variable, desc_ref)
    init_T = recc_acces(variable, init_T)

    if init_T.size(0) != 1:
        raise NotImplementedError('No implementation of batched ICP')
    if inv_init_T:
        init_T = init_T[0, :].inverse().unsqueeze(0)

    ICP_out = ICP.ICPwNet(pc_to_align, pc_ref, desc_to_align, desc_ref, init_T,
                    desc_function=(nets[0] if len(nets)>1 else None),
                    match_function=(nets[1] if len(nets)>1 else nets[0]),
                    pose_function=RSCPose.ransac_pose_estimation,
                          #pose_function=ICP.PoseFromMatching,
                    **param_icp)

    if inv_init_T:
        ICP_out['T'] = ICP_out['T'][0, :].inverse().unsqueeze(0)

    if filter_inliers:
        if ICP_out['inliers'] <= filter_inliers:
            ICP_out['T'] = init_T
    if filter_score:
        if ICP_out['score'] <= filter_score:
            ICP_out['T'] = init_T

    return {'T': ICP_out['T'],
            'q': utils.rot_to_quat(ICP_out['T'][0,:3,:3]).unsqueeze(0),
            'p': ICP_out['T'][0, :3, 3].unsqueeze(0),
            'score': ICP_out['score']}


def advanced_local_map_getter(nets, variable, **kwargs):
    Ts = kwargs.pop('T', False)
    descriptors_size = kwargs.pop('descriptors_size', 32)
    map_args = kwargs.pop('map_args', dict())

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    Ts = recc_acces(variable, Ts)
    size_pc = map_args.get('output_size', 2000)
    cnn_descriptor = map_args.get('cnn_descriptor', False)

    if not isinstance(size_pc, bool):
        batched_local_maps = Ts.new_zeros(Ts.size(0), 4, size_pc)
        if cnn_descriptor:
            encoders = Ts.new_zeros(Ts.size(0), descriptors_size, size_pc)
    elif Ts.size(0) != 1:
        raise AttributeError('Can generate full pc when batch size != 1.')

    for i, T in enumerate(Ts):
        if cnn_descriptor:
            if not isinstance(size_pc, bool):
                batched_local_maps[i], encoders[i] = utils.get_local_map(T=T, **map_args, cnn_enc=nets[0], cnn_dec=nets[1])
            else:
                batched_local_maps, encoders = utils.get_local_map(T=T, **map_args, cnn_enc=nets[0],
                                                                         cnn_dec=nets[1])
                batched_local_maps = batched_local_maps.unsqueeze(0)
                encoders = encoders.unsqueeze(0)
        else:
            if isinstance(size_pc, int):
                batched_local_maps[i] = utils.get_local_map(T=T, **map_args, cnn_enc=nets[0], cnn_dec=nets[1])
            else:
                batched_local_maps = utils.get_local_map(T=T, **map_args, cnn_enc=nets[0], cnn_dec=nets[1])
                batched_local_maps = batched_local_maps.unsqueeze(0)

    if cnn_descriptor:
        return {'pc': batched_local_maps, 'desc': encoders}
    else:
        return batched_local_maps


def batched_local_map_getter(variable, **kwargs):
    Ts = kwargs.pop('T', False)
    map_args = kwargs.pop('map_args', dict())

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    Ts = recc_acces(variable, Ts)
    size_pc = map_args.get('output_size', 2000)

    if isinstance(size_pc, int):
        batched_local_maps = Ts.new_zeros(Ts.size(0), 4, size_pc)

    for i, T in enumerate(Ts):
        if isinstance(size_pc, int):
            batched_local_maps[i] = utils.get_local_map(T=T, **map_args)
        else:
            batched_local_maps = utils.get_local_map(T=T, **map_args).unsqueeze(0)

    return batched_local_maps


def multi_batched_depth_map_to_pc(variable, **kwargs):
    depth_maps = kwargs.pop('depth_maps', None)
    K = kwargs.pop('K', None)
    inverse_depth = kwargs.pop('inverse_depth', True)
    remove_zeros = kwargs.pop('remove_zeros', False)
    eps = kwargs.pop('eps', 1e-8)
    scale_factor = kwargs.pop('scale_factor', None)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    batched_depth_maps = recc_acces(variable, depth_maps)
    num_pc = len(batched_depth_maps)
    params = {
        'scale_factor': scale_factor,
        'eps': eps,
        'remove_zeros': remove_zeros,
        'inverse_depth': inverse_depth
    }
    if remove_zeros:
        batched_pc = [batched_depth_map_to_pc(variable,
                                              depth_maps=depth_maps + [i],
                                              K=K + [i],
                                              **params)['pc'] for i in range(num_pc)]
    else:
        batched_pc = [batched_depth_map_to_pc(variable,
                                              depth_maps=depth_maps + [i],
                                              K=K + [i],
                                              **params) for i in range(num_pc)]
    return batched_pc


def batched_depth_map_to_pc(variable, **kwargs):
    depth_maps = kwargs.pop('depth_maps', None)
    K = kwargs.pop('K', None)
    inverse_depth = kwargs.pop('inverse_depth', True)
    remove_zeros = kwargs.pop('remove_zeros', False)
    eps = kwargs.pop('eps', 1e-8)
    scale_factor = kwargs.pop('scale_factor', None)
    multiples_instance = kwargs.pop('multiples_instance', False)
    modify_K = kwargs.pop('modify_K', True)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    batched_depth_maps = recc_acces(variable, depth_maps)

    if multiples_instance:
        batched_pc = [batched_depth_map_to_pc(variable,
                                              depth_maps=depth_maps + [i],
                                              K=K,
                                              inverse_depth=inverse_depth,
                                              remove_zeros=remove_zeros,
                                              eps=eps,
                                              scale_factor=scale_factor,
                                              multiples_instance=False,
                                              modify_K=False) for i in range(len(batched_depth_maps))]
    else:
        if modify_K:
            K = recc_acces(variable, K)
            if scale_factor is not None:
                if (1 / scale_factor) % 2 != 0:
                    raise ValueError('Scale factor is not a multiple of 2 (1/scale is {})'.format(1 / scale_factor))
                K[:, :2, :] *= scale_factor
        else:
            K = recc_acces(variable, K).clone().detach()

        if scale_factor is not None:
            batched_depth_maps = nn_func.interpolate(batched_depth_maps, scale_factor=scale_factor, mode='nearest')

        n_batch, _, height, width = batched_depth_maps.size()
        if remove_zeros:
            if n_batch != 1:
                raise ArithmeticError("Can't stack pc when removing zeros values! (batch_size!=1)")
        else:
            batched_pc = batched_depth_maps.new_zeros(n_batch, 4, height*width)

        for i, depth_maps in enumerate(batched_depth_maps):
            if inverse_depth:
                depth_maps = torch.reciprocal(depth_maps.clamp(min=eps)) - 1
            if remove_zeros:
                batched_pc, indexor = utils.depth_map_to_pc(depth_maps, K[i], remove_zeros)
                batched_pc = {'pc': batched_pc.unsqueeze(0), 'index': indexor}
            else:
                batched_pc[i, :, :] = utils.depth_map_to_pc(depth_maps, K[i], remove_zeros)

    return batched_pc

def index(variable, **kwargs):
    inputs = kwargs.pop('inputs', None)
    index = kwargs.pop('index', None)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    inputs = recc_acces(variable, inputs)
    index = recc_acces(variable, index)
    inputs = inputs.view(1, inputs.size(1), -1)
    inputs = inputs[:, :, index]

    return inputs

def resize(variable, **kwargs):
    inputs = kwargs.pop('inputs', None)
    scale_factor = kwargs.pop('scale_factor', None)
    flatten = kwargs.pop('flatten', True)
    mode = kwargs.pop('mode', 'nearest')
    multiples_instance = kwargs.pop('multiples_instance', False)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    if multiples_instance:
        inputs_var = recc_acces(variable, inputs[:1]) # Special treatment because of the double recc access
        inputs = [recc_acces(variable, inputs[:1] + [i] + inputs[1:]) for i in range(len(inputs_var))]
        n_batch = inputs[0].size(0)
        inputs = torch.cat(inputs, dim=0)
        if mode == 'bilinear':
            inputs = nn_func.interpolate(inputs, scale_factor=scale_factor, mode=mode, align_corners=True)
        else:
            inputs = nn_func.interpolate(inputs, scale_factor=scale_factor, mode=mode)
        if flatten:
            inputs = inputs.view(inputs.size(0),inputs.size(1), -1)
        return torch.split(inputs, n_batch, dim=0)

    else:
        inputs = recc_acces(variable, inputs)
        if mode == 'bilinear':
            inputs = nn_func.interpolate(inputs, scale_factor=scale_factor, mode=mode, align_corners=True)
        else:
            inputs = nn_func.interpolate(inputs, scale_factor=scale_factor, mode=mode)
        if flatten:
            inputs = inputs.view(inputs.size(0),inputs.size(1), -1)

    return inputs


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

    batched_pruned_pc = batched_pc.new_zeros(n_batch, 4, new_n_pt)
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


def matmul(variables, **kwargs):
    m1 = kwargs.pop('m1', None)
    m2 = kwargs.pop('m2', None)
    get_pq = kwargs.pop('get_pq', False)
    inv_m2 = kwargs.pop('inv_m2', False)
    inv_m1 = kwargs.pop('inv_m1', False)
    multiple_instances = kwargs.pop('multiple_instances', False)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    m1 = recc_acces(variables, m1)
    m2 = recc_acces(variables, m2)

    if multiple_instances:
        n_batch = m1[0].size(0)
        m1 = torch.cat(m1, dim=0)
        m2 = torch.cat(m2, dim=0)

    if inv_m1:
        m1 = torch.inverse(m1)
    if inv_m2:
        m2 = torch.inverse(m2)

    T = torch.matmul(m1, m2)
    if get_pq:
        return {'T': T,
                'p': T[:, :3, 3],
                'q': torch.cat([utils.rot_to_quat(Ti[:3, :3]).unsqueeze(0) for Ti in T], 0)}
    else:
        if multiple_instances:
            T = torch.split(T, n_batch, dim=0)
        return T

def batched_icp_desc(variable, **kwargs):
    batched_pc_ref = kwargs.pop('pc_ref', None)
    batched_desc_ref = kwargs.pop('desc_ref', None)
    batched_pc_to_align = kwargs.pop('pc_to_align', None)
    batched_desc_to_align = kwargs.pop('desc_to_align', None)
    batched_init_T = kwargs.pop('init_T', None)
    param_icp = kwargs.pop('param_icp', dict())
    detach_init_pose = kwargs.pop('detach_init_pose', False)
    desc = kwargs.pop('desc', False)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    batched_pc_ref = recc_acces(variable, batched_pc_ref)
    batched_pc_to_align = recc_acces(variable, batched_pc_to_align)
    if desc:
        batched_desc_ref = recc_acces(variable, batched_desc_ref)
        batched_desc_to_align = recc_acces(variable, batched_desc_to_align)

    batched_init_T = recc_acces(variable, batched_init_T)
    if detach_init_pose:
        batched_init_T = batched_init_T.detach()
        logger.debug('Detatching init pose')

    n_batch = batched_pc_to_align.size(0)
    poses = {
        'p': batched_pc_to_align.new_zeros(n_batch, 3),
        'q': batched_pc_to_align.new_zeros(n_batch, 4),
        'T': batched_pc_to_align.new_zeros(n_batch, 4, 4),
    }

    for i, pc_to_align in enumerate(batched_pc_to_align):
        args = list()
        if desc:
            args = [batched_desc_to_align[i], batched_desc_ref[i]]

        computed_pose, _ = ICP.ICPwNet(pc_to_align,
                                       batched_pc_ref[i],
                                       batched_init_T[i],
                                       *args,
                                       **param_icp)
        poses['p'][i, :] = computed_pose[:3, 3]
        poses['q'][i, :] = utils.rot_to_quat(computed_pose[:3, :3])
        poses['T'][i, :, :] = computed_pose

    return poses

def batched_icp(variable, **kwargs):
    batched_pc_ref = kwargs.pop('pc_ref', None)
    batched_pc_to_align = kwargs.pop('pc_to_align', None)
    batched_ref = kwargs.pop('batched_ref', True)
    batched_init_T = kwargs.pop('init_T', None)
    T_gt = kwargs.pop('T_gt', None)
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
    if T_gt:
        T_gt = recc_acces(variable, T_gt)

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
        T_gt_i = T_gt[i] if T_gt is not None else None
        if batched_ref:
            computed_pose, d = ICP.soft_icp(batched_pc_ref[i, :, :], pc_to_align, init_T, **param_icp, custom_filter=current_filter, T_gt=T_gt_i)
        else:
            computed_pose, d = ICP.soft_icp(batched_pc_ref, pc_to_align, init_T, **param_icp, custom_filter=current_filter, T_gt=T_gt_i)
        dist[i] = d
        poses['p'][i, :] = computed_pose[:3, 3]
        poses['q'][i, :] = utils.rot_to_quat(computed_pose[:3, :3])
        poses['T'][i, :, :] = computed_pose

    return {'errors': dist, 'poses': poses}

