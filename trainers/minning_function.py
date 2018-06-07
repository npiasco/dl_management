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


def default(trainer, batch, mode):
    return trainer.network(auto.Variable(trainer.cuda_func(batch[mode][trainer.mod]), requires_grad=True))


def random(trainer, batch, mode, return_idx=False):
    n = len(batch[mode])
    pick = rd.randint(0, n-1)
    if return_idx:
        return trainer.network(auto.Variable(trainer.cuda_func(batch[mode][pick][trainer.mod]), requires_grad=True)),\
        [pick for i in range(batch[mode][0][trainer.mod].size(0))]
    else:
        return trainer.network(auto.Variable(trainer.cuda_func(batch[mode][pick][trainer.mod]), requires_grad=True))


def hard_minning(trainer, batch, mode, **kwargs):
    return_idx = kwargs.pop('return_idx', False)
    n_ex = kwargs.pop('n_ex', {'positives': 2, 'negatives': 5})
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    n_ex = n_ex[mode]

    trainer.network.eval()
    anchors = trainer.network(auto.Variable(trainer.cuda_func(batch['query'][trainer.mod]),
                                            requires_grad=False))
    desc_anchors = anchors
    size = batch['query'][trainer.mod].size()
    size = torch.Size((n_ex, size[0], size[1], size[2], size[3]))
    exemples = torch.FloatTensor(size)
    idxs = [list() for _ in range(n_ex)]
    for i, desc_anchor in enumerate(desc_anchors):
        ex_descs = [trainer.network(auto.Variable(trainer.cuda_func(ex[trainer.mod][i:i+1]),
                                                  requires_grad=False))
                    for ex in batch[mode]]
        diff = [func.pairwise_distance(desc_anchor.unsqueeze(0), x).data.cpu().numpy()[0, 0] for x in ex_descs]
        sort_index = np.argsort(diff)
        if mode == 'positives':
            idx = sort_index[-1*n_ex:]
        elif mode == 'negatives':
            idx = sort_index[:n_ex]
        for j in range(n_ex):
            exemples[j][i] = batch[mode][idx[j]][trainer.mod][i]
            idxs[j].append(idx[j])

    trainer.network.train()
    # Forward
    forwarded_ex = None
    for ex in exemples:
        forward = trainer.network(auto.Variable(trainer.cuda_func(ex), requires_grad=True))
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

@static_vars(dataset=None)
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

    if hard_mining_augmented.dataset is None:
        env_var = os.environ[neg_pool['env_var']]
        hard_mining_augmented.dataset = system.BaseClass.Base.creat_dataset(neg_pool['dataset'], env_var)

    dload = torch.utils.data.DataLoader(hard_mining_augmented.dataset, **neg_pool['loader_param'])
    dload = dload.__iter__()
    random_batch = {
        'query': batch['query'],
        'negatives': [dload.__next__() for i in range(neg_pool['num_ex'])]
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
