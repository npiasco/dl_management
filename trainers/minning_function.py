import setlog
import torch.autograd as auto
import random as rd
import torch.nn.functional as func
import numpy as np
import torch
import sklearn.neighbors as skn


logger = setlog.get_logger(__name__)


def recc_acces(var, names):
    if not names:
        return var
    else:
        sub_name = names[1:]
        return recc_acces(var[names[0]], sub_name)


def images_from_poses(variable, **kwargs):
    poses = kwargs.pop('poses', None)
    data = kwargs.pop('data', None)
    mode = kwargs.pop('mode', 'rgb')

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    poses = recc_acces(variable, poses)
    data = recc_acces(variable, data)

    if not hasattr(images_from_poses, 'nn_computor'):
        logger.info('New nn indexing, fitting the db...')
        data_poses = [data.get_position(i).reshape(1, 3) for i in range(len(data))]

        images_from_poses.nn_computor = skn.NearestNeighbors(n_neighbors=1)
        images_from_poses.nn_computor.fit(np.concatenate(data_poses))

    if isinstance(poses, list):
        batchs = list()
        for pose in poses:
            if isinstance(pose, dict):
                dev = pose['p'].device
                target_pose = pose['p']
            else:
                dev = pose.device
                target_pose = pose[0, :3, 3].view(1, 3)

            idx = images_from_poses.nn_computor.kneighbors(target_pose.cpu().numpy(), return_distance=False)

            batchs.append(data[idx[0, 0]][mode].unsqueeze(0).to(dev))
    else:
        idx = images_from_poses.nn_computor.kneighbors(poses['p'].cpu().numpy(), return_distance=False)
        batchs = data[idx[0, 0]][mode].unsqueeze(0).to(poses['p'].device)

    return batchs


def get_nn_pose_from_desc(variable, **kwargs):
    feat = kwargs.pop('feat', None)
    db = kwargs.pop('db', None)
    metric = kwargs.pop('metric', 'cosine')
    k_nn = kwargs.pop('k_nn', 1)
    step_k_nn = kwargs.pop('step_k_nn', 1)
    angle_threshold = kwargs.pop('angle_threshold', None) # degree
    pos_threshold = kwargs.pop('pos_threshold', None) # m
    return_only_T = kwargs.pop('return_only_T', False) # m

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    feats = recc_acces(variable, feat)
    db = recc_acces(variable, db)

    if not hasattr(get_nn_pose_from_desc, 'nn_computor'):
        logger.info('New nn indexing, fitting the db...')
        get_nn_pose_from_desc.nn_computor = skn.NearestNeighbors(n_neighbors=1, metric=metric)
        get_nn_pose_from_desc.nn_computor.fit(torch.stack(db['feat']).cpu().numpy())

    idx = get_nn_pose_from_desc.nn_computor.kneighbors(feats.cpu().numpy(), return_distance=False, n_neighbors=k_nn*step_k_nn)

    if k_nn > 1:
        if angle_threshold:
            poses = [db['pose'][idx[0, 0]]]
            idx_next = 1
            curr_d_angle = float('inf')
            for i in range(1, k_nn*step_k_nn):
                d_angle = 180/3.14159265 * 2 * torch.acos(
                    torch.abs(
                        torch.dot(db['pose'][idx[0, 0]]['q'][0], db['pose'][idx[0, i]]['q'][0]))
                )
                if d_angle < angle_threshold and (d_angle > curr_d_angle or curr_d_angle == float('inf')):
                    curr_d_angle = d_angle
                    idx_next = i
                elif d_angle > angle_threshold and (d_angle < curr_d_angle or curr_d_angle < angle_threshold):
                    curr_d_angle = d_angle
                    idx_next = i
            poses.append(db['pose'][idx[0, idx_next]])
            return poses
        elif pos_threshold:
            poses = [db['pose'][idx[0, 0]]]
            idx_next = 1
            curr_d_pos = float('inf')
            for i in range(1, k_nn*step_k_nn):
                d_pos = torch.sqrt(
                    torch.sum(
                        (db['pose'][idx[0, 0]]['p'][0] - db['pose'][idx[0, i]]['p'][0])**2
                    )
                )
                if d_pos < pos_threshold and (d_pos > curr_d_pos or curr_d_pos == float('inf')):
                    curr_d_pos = d_pos
                    idx_next = i
                elif d_pos > pos_threshold and (d_pos < curr_d_pos or curr_d_pos < pos_threshold):
                    curr_d_pos = d_pos
                    idx_next = i
            poses.append(db['pose'][idx[0, idx_next]])
            return poses

        else:
            if return_only_T:
                return [db['pose'][idx[0, i]]['T'] for i in range(0, k_nn * step_k_nn, step_k_nn)]
            else:
                return [db['pose'][idx[0, i]] for i in range(0, k_nn*step_k_nn, step_k_nn)]
    else:
        if return_only_T:
            return db['pose'][idx[0, 0]]['T']
        else:
            return db['pose'][idx[0, 0]]


def construct_feat_database(variable, **kwargs):
    feat = kwargs.pop('feat', None)
    pose = kwargs.pop('pose', None)
    db = kwargs.pop('db', None)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    feats  = recc_acces(variable, feat)
    pose = recc_acces(variable, pose)
    try:
        db = recc_acces(variable, db)
    except KeyError:
        db = {'feat': list(), 'pose': list()}

    db['feat'].append(feats.squeeze(0).cpu())
    #TODO: pose on cpu as well (but problem with dico)
    db['pose'].append(pose)

    return db


def outliers_count(variable, **kwargs):
    pc_ref = kwargs.pop('pc_ref', None)
    pc_to_align = kwargs.pop('pc_to_align', None)
    T = kwargs.pop('T', None)
    sigma = kwargs.pop('sigma', 1e-1)
    beta = kwargs.pop('beta', 50)

    pc_ref = recc_acces(variable, pc_ref)
    pc_to_align = recc_acces(variable, pc_to_align)
    T = recc_acces(variable, T)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    return torch.sum(
        torch.sigmoid(
            beta * (
                torch.sum(
                    (pc_ref - T.matmul(pc_to_align)) ** 2,
                    1)
                - sigma
            )
        )
    )/(pc_ref.size(0)*pc_ref.size(-1))


def default(network, batch, mode, **kwargs):
    cuda_func = kwargs.pop('cuda_func', lambda x: x.cuda())
    mod = kwargs.pop('mod', None)
    return_idx = kwargs.pop('return_idx', False)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    return network(auto.Variable(cuda_func(batch[mode][mod]), requires_grad=True))


def random(network, batch, mode, **kwargs):
    cuda_func = kwargs.pop('cuda_func', lambda x: x.cuda())
    mod = kwargs.pop('mod', None)
    return_idx = kwargs.pop('return_idx', False)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    n = len(batch[mode])
    pick = rd.randint(0, n-1)
    if return_idx:
        return network(auto.Variable(cuda_func(batch[mode][pick][mod]), requires_grad=True)),\
        [pick for i in range(batch[mode][0][mod].size(0))]
    else:
        return network(auto.Variable(cuda_func(batch[mode][pick][mod]), requires_grad=True))


def hard_minning(network, batch, mode, **kwargs):
    return_idx = kwargs.pop('return_idx', False)
    n_ex = kwargs.pop('n_ex', {'positives': 2, 'negatives': 5})
    cuda_func = kwargs.pop('cuda_func', lambda x: x.cuda())
    mod = kwargs.pop('mod', None)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    n_ex = n_ex[mode]

    network.eval()
    anchors = network(auto.Variable(cuda_func(batch['query'][mod]),
                                    requires_grad=False))
    desc_anchors = anchors
    size = batch['query'][mod].size()
    size = torch.Size((n_ex, size[0], size[1], size[2], size[3]))
    exemples = torch.FloatTensor(size)
    idxs = [list() for _ in range(n_ex)]
    for i, desc_anchor in enumerate(desc_anchors):
        ex_descs = [network(auto.Variable(cuda_func(ex[mod][i:i+1]),
                                          requires_grad=False))
                    for ex in batch[mode]]
        diff = [func.pairwise_distance(desc_anchor.unsqueeze(0), x).data.cpu().numpy()[0, 0] for x in ex_descs]
        sort_index = np.argsort(diff)
        if mode == 'positives':
            idx = sort_index[-1*n_ex:]
        elif mode == 'negatives':
            idx = sort_index[:n_ex]
        for j in range(n_ex):
            exemples[j][i] = batch[mode][idx[j]][mod][i]
            idxs[j].append(idx[j])

    network.train()
    # Forward
    forwarded_ex = None
    for ex in exemples:
        forward = network(auto.Variable(cuda_func(ex), requires_grad=True))
        if forwarded_ex is None:
            forwarded_ex = dict()
            for name, val in forward.items():
                # TODO: improve this part
                if isinstance(val, dict):
                    forwarded_ex[name] = dict()
                    for name_2, val_2 in val.items():
                        forwarded_ex[name][name_2] = list()
                        forwarded_ex[name][name_2].append(val_2)
                else:
                    forwarded_ex[name] = list()
                    forwarded_ex[name].append(val)
        else:
            for name, val in forward.items():
                if isinstance(val, dict):
                    for name_2, val_2 in val.items():
                        forwarded_ex[name][name_2].append(val_2)
                else:
                    forwarded_ex[name].append(val)

    if return_idx:
        return forwarded_ex, idxs
    else:
        return forwarded_ex


def no_selection(trainer, batch, mode):
    exemples = None
    for ex in batch[mode]:
        forward = trainer.network(auto.Variable(trainer.cuda_func(ex[trainer.mod]), requires_grad=True))
        if exemples is None:
            exemples = dict()
            for name, val in forward.items():
                if isinstance(val, dict):
                    exemples[name] = dict()
                    for name_2, val_2 in val.items():
                        exemples[name][name_2] = list()
                        exemples[name][name_2].append(val_2)
                else:
                    exemples[name] = list()
                    exemples[name].append(val)
        else:
            for name, val in forward.items():
                if isinstance(val, dict):
                    for name_2, val_2 in val.items():
                        exemples[name][name_2].append(val_2)
                else:
                    exemples[name].append(val)

    return exemples


def batch_forward(net, batch, **kwargs):
    mode = kwargs.pop('mode', None)
    target = kwargs.pop('target', None)
    cuda_func = kwargs.pop('cuda_func', lambda x: x.cuda())

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    batch = batch['batch']

    if mode == 'query':
        forward = net(auto.Variable(cuda_func(recc_acces(batch, target)), requires_grad=False))
    else:
        forward = dict()
        for sub_batch in batch[mode]:
            outputs = net(auto.Variable(cuda_func(recc_acces(sub_batch, target)), requires_grad=False))
            for name, output in outputs.items():
                if name in forward.keys():
                    forward[name].append(output)
                else:
                    forward[name] = [output]

    return forward


def detach_input(variables, **kwargs):
    inputs = kwargs.pop('inputs', None)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    inputs = recc_acces(variables, inputs)
    inputs = inputs.detach()

    return inputs


def batch_to_var(net, batch, **kwargs):
    # TODO: Automatically inside the training loop
    mode = kwargs.pop('mode', None)
    target = kwargs.pop('target', None)
    mult_mod = kwargs.pop('mult_mod', False)
    cuda_func = kwargs.pop('cuda_func', lambda x: x.cuda())

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    batch = batch['batch']

    if mode == 'query':
        if mult_mod:
            forward = dict()
            for name_mod, mod in recc_acces(batch, target).items():
                forward[name_mod] = mod
        else:
            forward = recc_acces(batch, target)
    else:
        if mult_mod:
            forward = dict()
            for sub_batch in batch[mode]:
                for name_mod, mod in recc_acces(sub_batch, target).items():
                    if name_mod in forward.keys():
                        forward[name_mod].append(mod)
                    else:
                        forward[name_mod] = [mod]
        else:
            forward = list()
            for sub_batch in batch[mode]:
                forward.append(recc_acces(sub_batch, target))

    return forward


def simple_multiple_forward(net, variables, **kwargs):
    input_targets = kwargs.pop('input_targets', list())

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)
    inputs = recc_acces(variables, input_targets)
    try:
        num_elem = len(inputs)
        b, _, _, _ = inputs[0].size()
        inputs = torch.cat(inputs, dim=0)
        forwarded = net(inputs)
        outputs = [dict() for _ in range(num_elem)]
        for name, val in forwarded.items():
            splited_vals = torch.split(val, b, dim=0)
            for i, splited_val in enumerate(splited_vals):
                outputs[i][name] = splited_val
    except AttributeError:
        outputs = [net(input) for input in inputs]

    return outputs


def custom_forward(net, outputs, **kwargs):
    input_targets = kwargs.pop('input_targets', list())
    multiples_instance = kwargs.pop('multiples_instance', False)
    detach_inputs = kwargs.pop('detach_inputs', False)
    is_dict = kwargs.pop('dict', False)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    inputs = [recc_acces(outputs, name) for name in input_targets]
    if multiples_instance:
        forward = dict()
        for i in range(len(inputs[0])):
            if detach_inputs:
                tmp_input = [inp[i].detach() for inp in inputs]
            else:
                tmp_input = [inp[i] for inp in inputs]
            forwarded = net(*tmp_input)
            if isinstance(forwarded, dict):
                for name_out, out in forwarded.items():
                    if name_out in forward.keys():
                        forward[name_out].append(out)
                    else:
                        forward[name_out] = [out]
            elif isinstance(forward, dict):
                forward = [forwarded]
            else:
                forward.append(forwarded)

    else:
        if detach_inputs:
            if is_dict:
                new_inputs = list()
                for inp in inputs:
                    for name, value in inp.items():
                        inp[name] = value.detach()
                    new_inputs.append(inp)
                inputs = new_inputs
            else:
                inputs = [inp.detach() for inp in inputs]

        forward = net(*inputs)

    return forward


def random_prunning(outputs, **kwargs):
    prob = kwargs.pop('prob', 0.5)
    target = kwargs.pop('target', list())
    replacement_value = kwargs.pop('replacement_value', 1)
    multiples_instance = kwargs.pop('multiples_instance', False)
    mask = kwargs.pop('mask', None)
    target_density = kwargs.pop('target_density', None)
    target_density_map = kwargs.pop('target_density_map', None)
    kernel_size = kwargs.pop('kernel_size', 25)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    input_var = [recc_acces(outputs, target)] if not multiples_instance else recc_acces(outputs, target)

    if mask is not None:
        mask_var = [recc_acces(outputs, mask)] if not multiples_instance else recc_acces(outputs, mask)
    if target_density is not None:
        target_density_var = [recc_acces(outputs, target_density)] if not multiples_instance else\
            recc_acces(outputs, target_density)
    if target_density_map is not None:
        target_d_map_var = [recc_acces(outputs, target_density_map)] if not multiples_instance else\
            recc_acces(outputs, target_density_map)

    pruned_input = list()
    for i, inst in enumerate(input_var):
        if mask is not None:
            indexor = mask_var[i].data != replacement_value
        elif target_density_map is not None:
            b, c, w, h = inst.size()
            indexor = torch.zeros(inst.size())

            for wi in range(0, w, kernel_size):
                for hi in range(0, h, kernel_size):
                    if wi + kernel_size > w:
                        if hi + kernel_size > h:
                            cropped = target_d_map_var[i][:, :, -kernel_size:, -kernel_size:]
                        else:
                            cropped = target_d_map_var[i][:, :, -kernel_size:, hi:hi + kernel_size]
                    else:
                        if hi + kernel_size > h:
                            cropped = target_d_map_var[i][:, :, wi:wi + kernel_size, -kernel_size:]
                        else:
                            cropped = target_d_map_var[i][:, :, wi:wi + kernel_size, hi:hi + kernel_size]

                    density = torch.numel(
                        torch.nonzero((cropped == replacement_value).view(-1))
                    )/(kernel_size**2)

                    prob_density = (torch.rand(b, c, kernel_size, kernel_size) > density).float()

                    if wi + kernel_size > w:
                        if hi + kernel_size > h:
                            indexor[:, :, -kernel_size:, -kernel_size:] = prob_density
                        else:
                            indexor[:, :, -kernel_size:, hi:hi + kernel_size] = prob_density
                    else:
                        if hi + kernel_size > h:
                            indexor[:, :, wi:wi + kernel_size, -kernel_size:] = prob_density
                        else:
                            indexor[:, :, wi:wi + kernel_size, hi:hi + kernel_size] = prob_density
        else:
            if target_density is not None:
                prob = (
                    torch.numel(
                        torch.nonzero(
                            (target_density_var[i].data == replacement_value).view(-1)
                        )
                    ) / torch.numel(target_density_var[i].data)
                )
            indexor = torch.rand(inst.size()) > prob
        indexor = auto.Variable(indexor.float(), requires_grad=False)
        if inst.is_cuda:
            indexor = indexor.cuda()
        tmp_pruned = inst*indexor + (replacement_value - indexor*replacement_value)
        pruned_input.append(tmp_pruned)

    if not multiples_instance:
        pruned_input = pruned_input[0]

    return pruned_input


def general_hard_minning(outputs, **kwargs):
    return_idx = kwargs.pop('return_idx', False)
    n_ex = kwargs.pop('n_ex', 1)
    anchor_getter = kwargs.pop('anchor_getter', list())
    example_getter = kwargs.pop('example_getter', list())
    mode = kwargs.pop('mode', None)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    desc_anchors = recc_acces(outputs, anchor_getter)
    examples = recc_acces(outputs, example_getter)
    idxs = [list() for _ in range(n_ex)]
    forwarded_ex = [None for _ in range(n_ex)]

    for i, desc_anchor in enumerate(desc_anchors):
        diff = [func.pairwise_distance(desc_anchor.unsqueeze(0), x[i:i+1]).item() for x in examples]
        sort_index = np.argsort(diff)
        if mode == 'positives':
            idx = sort_index[-1*n_ex:]
        elif mode == 'negatives':
            idx = sort_index[:n_ex]
        for j in range(n_ex):
            idxs[j].append(idx[j])
            if forwarded_ex[j] is None:
                forwarded_ex[j] = examples[idx[j]][i:i + 1]
            else:
                forwarded_ex[j] = torch.cat((forwarded_ex[j], examples[idx[j]][i:i + 1]))

    if return_idx:
        return {'ex': forwarded_ex, 'idx': idxs}
    else:
        return forwarded_ex


def examples_selection(outputs, **kwargs):
    idxs = kwargs.pop('idxs', list())
    getter = kwargs.pop('getter', list())

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    examples = recc_acces(outputs, getter)

    idxs = recc_acces(outputs, idxs)
    selected_exemples = [None for _ in range(len(idxs))]
    for batch_num, l_i in enumerate(idxs):
        for i, j in enumerate(l_i):
            if selected_exemples[batch_num] is None:
                selected_exemples[batch_num] = examples[j][i].unsqueeze(0)
            else:
                selected_exemples[batch_num] = torch.cat((selected_exemples[batch_num], examples[j][i].unsqueeze(0)), dim=0)

    return selected_exemples
