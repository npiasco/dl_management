import setlog
import torch.optim as optim
import copy
import torch


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
        self.best_net = (None, copy.deepcopy(self.network.state_dict()))
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
            return x.cpu()

    def train(self, batch):
        raise NotImplementedError()

    def eval(self, **kwargs):
        raise NotImplementedError()

    def test(self, **kwargs):
        raise NotImplementedError()

    def serialize(self):
        self.network.cpu()
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cpu()

        ser = {
            'network': copy.deepcopy(self.network.state_dict()),
            'best_network': copy.deepcopy(self.best_net),
            'optimizer': copy.deepcopy(self.optimizer.state_dict()),
            'loss': copy.deepcopy(self.loss_log),
            'val_score': copy.deepcopy(self.val_score)
        }

        self.cuda_func(self.network)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = self.cuda_func(v)

        return ser

    def load(self, datas):
        self.network.cpu()
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cpu()

        self.network.load_state_dict(datas['network'])
        self.best_net = datas['best_network']
        self.loss_log = datas['loss']
        self.val_score = datas['val_score']

        self.cuda_func(self.network)

        self.optimizer = self.init_optimizer(
            self.network.get_training_layers()
        )

        self.optimizer.load_state_dict(datas['optimizer'])

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = self.cuda_func(v)
