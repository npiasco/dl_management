import setlog
import torch.optim as optim


logger = setlog.get_logger(__name__)


class BaseTrainer:
    def __init__(self, **kwargs):
        self.batch_size = kwargs.pop('batch_size', 25)
        self.max_epoch = kwargs.pop('max_epoch', 100)
        self.lr = kwargs.pop('lr', 0.005)
        self.momentum = kwargs.pop('momentum', 0.9)
        self.weight_decay = kwargs.pop('weight_decay', 0.001)
        self.shuffle = kwargs.pop('shuffle', True)
        self.cuda_on = kwargs.pop('cuda_on', True)
        self.optimizer_type = kwargs.pop('optimizer_type', 'SGD')
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

    def init_optimizer(self, param):
        if self.optimizer_type == "SGD":
            optimizer = optim.SGD(param, lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
        elif self.optimizer_type == "ADAM":
            optimizer = optim.Adam(param, lr=self.lr, weight_decay=self.weight_decay)
        else:
            logger.error('Unknown optimizer {}'.format(self.optimizer_type))
            optimizer = None

        return optimizer

    def cuda_func(self, x):
        if self.cuda_on:
            return x.cuda()
        else:
            return x

    def train(self, batch):
        raise NotImplementedError()

    def eval(self):
        raise NotImplementedError()

    def serialize(self):
        raise NotImplementedError()
