import setlog
import torch.autograd as auto
import random as rd


logger = setlog.get_logger(__name__)


def default(trainer, batch, mode):
    return trainer.network(auto.Variable(trainer.cuda_func(batch[mode][trainer.mod]), requires_grad=True))


def random(trainer, batch, mode):
    n = len(batch[mode])
    pick = rd.randint(0, n-1)
    return trainer.network(auto.Variable(trainer.cuda_func(batch[mode][pick][trainer.mod]), requires_grad=True))


def hard_minning(trainer, batch, mode):
    raise NotImplementedError


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
