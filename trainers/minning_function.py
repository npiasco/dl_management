import setlog
import torch.autograd as auto
import random as rd
import torch.nn.functional as func
import numpy as np
import torch


logger = setlog.get_logger(__name__)


def default(trainer, batch, mode):
    return trainer.network(auto.Variable(trainer.cuda_func(batch[mode][trainer.mod]), requires_grad=True))


def random(trainer, batch, mode):
    n = len(batch[mode])
    pick = rd.randint(0, n-1)
    return trainer.network(auto.Variable(trainer.cuda_func(batch[mode][pick][trainer.mod]), requires_grad=True))


def hard_minning(trainer, batch, mode):
    anchors = trainer.network(auto.Variable(trainer.cuda_func(batch['query'][trainer.mod]),
                                            requires_grad=False))
    desc_anchors = anchors['desc']
    exemples = torch.FloatTensor(batch['query'][trainer.mod].size())
    trainer.network.eval()
    for i, desc_anchor in enumerate(desc_anchors):
        ex_descs = [trainer.network(auto.Variable(trainer.cuda_func(ex[trainer.mod][i:i+1]),
                                                  requires_grad=False))['desc']
                    for ex in batch[mode]]
        diff = [func.pairwise_distance(desc_anchor.unsqueeze(0), x).data.cpu().numpy()[0, 0] for x in ex_descs]
        sort_index = np.argsort(diff)
        if mode == 'positives':
            exemples[i] = batch[mode][sort_index[-1]][trainer.mod][i]
        elif mode == 'negatives':
            exemples[i] = batch[mode][sort_index[0]][trainer.mod][i]
    trainer.network.train()
    return trainer.network(auto.Variable(trainer.cuda_func(exemples), requires_grad=True))


def no_selection(trainer, batch, mode):
    exemples = {
        'desc': list(),
        'feat': list()
    }
    for ex in batch[mode]:
        forward = trainer.network(auto.Variable(trainer.cuda_func(ex[trainer.mod]), requires_grad=True))
        exemples['desc'].append(forward['desc'])
        exemples['feat'].append(forward['feat'])

    return exemples
