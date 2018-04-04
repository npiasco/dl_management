import setlog
import torch.autograd as auto
import random as rd


logger = setlog.get_logger(__name__)


def default(trainer, batch, mode):
    return trainer.network(trainer.cuda_func(auto.Variable(batch[mode][trainer.mod], requires_grad=True)))


def random(trainer, batch, mode):
    n = len(batch[mode])
    pick = rd.randint(0, n-1)
    return trainer.network(trainer.cuda_func(auto.Variable(batch[mode][pick][trainer.mod], requires_grad=True)))

def hard_minning(trainer, batch, mode):
    raise NotImplementedError

