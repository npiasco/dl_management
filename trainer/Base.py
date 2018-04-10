import setlog
import torch.optim as optim
import copy

logger = setlog.get_logger(__name__)


class BaseTrainer:
    def __init__(self, **kwargs):
        self.lr = kwargs.pop('lr', 0.005)
        self.momentum = kwargs.pop('momentum', 0.9)
        self.weight_decay = kwargs.pop('weight_decay', 0.001)
        self.cuda_on = kwargs.pop('cuda_on', True)
        self.optimizer_type = kwargs.pop('optimizer_type', 'SGD')
        self.network = kwargs.pop('network', None)
        self.val_num_workers = kwargs.pop('val_num_workers', 8)
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        self.val_score = list()
        self.loss_log = dict()

        self.network.cpu()
        self.best_net = (0, copy.deepcopy(self.network.state_dict()))
        self.cuda_func(self.network)
        self.optimizer = None

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

    def eval(self, queries, dataset, score_function, ep):
        raise NotImplementedError()

    def test(self, queries, dataset, score_functions):
        raise NotImplementedError()

    def serialize(self):
        ser = {
            'network': self.network.state_dict(),
            'best_network': self.best_net,
            'optimizer': self.optimizer.state_dict(),
            'loss': self.loss_log,
            'val_score': self.val_score
        }

        return ser

    def load(self, datas):
        self.network.load_state_dict(datas['network'])
        self.best_net = datas['best_network']
        self.optimizer.load_state_dict(datas['optimizer'])
        self.loss_log = datas['loss']
        self.val_score = datas['val_score']
