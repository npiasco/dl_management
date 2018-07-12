import setlog
import torch.autograd as auto
import random as rd
import torch.nn.functional as func
import numpy as np
import torch
import system
import os
import copy


logger = setlog.get_logger(__name__)


def static_vars(**kwargs):
    def decorate(func):
        for k, v in kwargs.items():
            setattr(func, k, v)
        return func
    return decorate


def default(network, batch, mode, **kwargs):
    cuda_func = kwargs.pop('cuda_func', None)
    mod = kwargs.pop('mod', None)
    return_idx = kwargs.pop('return_idx', False)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    return network(auto.Variable(cuda_func(batch[mode][mod]), requires_grad=True))


def random(network, batch, mode, **kwargs):
    cuda_func = kwargs.pop('cuda_func', None)
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
    cuda_func = kwargs.pop('cuda_func', None)
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

@static_vars(dload=None)
def hard_mining_augmented(trainer, batch, mode, **kwargs):
    ckwargs = copy.deepcopy(kwargs)
    neg_pool = ckwargs.pop('neg_pool', None)

    if mode == 'positives':
        return hard_minning(trainer, batch, mode, **ckwargs)
    else:
        if ckwargs.get('return_idx', False):
            forwarded_ex, idxs = hard_minning(trainer, batch, mode, **ckwargs)
        else:
            forwarded_ex = hard_minning(trainer, batch, mode, **ckwargs)

    ckwargs = copy.deepcopy(kwargs)
    neg_pool = ckwargs.pop('neg_pool', None)

    if hard_mining_augmented.dload is None:
        env_var = os.environ[neg_pool['env_var']]
        dataset = system.BaseClass.Base.creat_dataset(neg_pool['dataset'], env_var)
        hard_mining_augmented.dload = torch.utils.data.DataLoader(dataset, **neg_pool['loader_param'])
        hard_mining_augmented.dload = hard_mining_augmented.dload.__iter__()

    try:
        random_batch = {
            'query': batch['query'],
            'negatives': [hard_mining_augmented.dload.__next__() for i in range(neg_pool['num_ex'])]
        }
    except StopIteration:
        logger.info("Restarting hard neg pool")
        hard_mining_augmented.dload = torch.utils.data.DataLoader(hard_mining_augmented.dload.dataset,
                                                                  **neg_pool['loader_param'])
        hard_mining_augmented.dload = hard_mining_augmented.dload.__iter__()
        random_batch = {
            'query': batch['query'],
            'negatives': [hard_mining_augmented.dload.__next__() for i in range(neg_pool['num_ex'])]
        }

    if ckwargs.get('return_idx', False):
        augmented_forwarded_ex, _ = hard_minning(trainer, random_batch, mode, **ckwargs)
    else:
        augmented_forwarded_ex = hard_minning(trainer, random_batch, mode, **ckwargs)

    for name, val in augmented_forwarded_ex.items():
        if isinstance(val, dict):
            for name_2, val_2 in val.items():
                forwarded_ex[name][name_2] += val_2
        else:
            forwarded_ex[name] += val

    if ckwargs.get('return_idx', False):
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
