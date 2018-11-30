import setlog
import torch.autograd as auto
import random as rd
import torch.nn.functional as func
import numpy as np
import torch


logger = setlog.get_logger(__name__)


def recc_acces(var, names):
    if not names:
        return var
    else:
        sub_name = names[1:]
        return recc_acces(var[names[0]], sub_name)


def outliers_count(variable, **kwargs):
    pc_ref = kwargs.pop('pc_ref', None)
    pc_to_align = kwargs.pop('pc_to_align', None)
    T = kwargs.pop('T', None)
    sigma = kwargs.pop('sigma', 1e-1)
    beta = kwargs.pop('beta', 0.5)

    pc_ref = recc_acces(variable, pc_ref)
    pc_to_align = recc_acces(variable, pc_to_align)
    T = recc_acces(variable, T)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    return torch.sum(
        torch.sigmoid(
            (
                beta * torch.sum(
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
                forward[name_mod] = auto.Variable(cuda_func(mod), requires_grad=False)
        else:
            forward = auto.Variable(cuda_func(recc_acces(batch, target)), requires_grad=False)
    else:
        if mult_mod:
            forward = dict()
            for sub_batch in batch[mode]:
                for name_mod, mod in recc_acces(sub_batch, target).items():
                    if name_mod in forward.keys():
                        forward[name_mod].append(auto.Variable(cuda_func(mod), requires_grad=False))
                    else:
                        forward[name_mod] = [auto.Variable(cuda_func(mod), requires_grad=False)]
        else:
            forward = list()
            for sub_batch in batch[mode]:
                forward.append(auto.Variable(cuda_func(recc_acces(sub_batch, target)), requires_grad=False))

    return forward


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
        diff = [func.pairwise_distance(desc_anchor.unsqueeze(0), x[i:i+1]).data.cpu().numpy()[0, 0] for x in examples]
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
