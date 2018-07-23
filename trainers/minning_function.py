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
        forward = net(auto.Variable(cuda_func(recc_acces(batch, target))))
    else:
        forward = dict()
        for sub_batch in batch[mode]:
            outputs = net(auto.Variable(cuda_func(recc_acces(sub_batch, target))))
            for name, output in outputs.items():
                if name in forward.keys():
                    forward[name].append(output)
                else:
                    forward[name] = [output]

    return forward


def custom_forward(net, outputs, **kwargs):
    input_targets = kwargs.pop('input_targets', list())
    multiples_instance = kwargs.pop('multiples_instance', False)
    cuda_func = kwargs.pop('cuda_func', lambda x: x.cuda())

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    inputs = [recc_acces(outputs, name) for name in input_targets]
    if multiples_instance:
        forward = list()
        for i in range(len(inputs[0])):
            tmp_input = [inp[i] for inp in inputs]
            forward.append(net(*tmp_input))
    else:
        forward = net(*inputs)

    return forward


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
