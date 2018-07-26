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

        if self.cuda_on is False:
            logger.warning('CUDA DISABLE, Training may take a while')

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


class BaseMultNetTrainer:
    def __init__(self, **kwargs):
        self.cuda_on = kwargs.pop('cuda_on', True)
        self.val_num_workers = kwargs.pop('val_num_workers', 8)
        self.optimizers_params = kwargs.pop('optimizers_params', None)
        self.networks = kwargs.pop('networks', None)

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        self.optimizers = self.init_optimizers(self.optimizers_params)

        self.val_score = list()
        self.loss_log = dict()
        self.best_net = list()
        self._save_current_net(None)

        if self.cuda_on is False:
            logger.warning('CUDA DISABLE, Training may take a while')

    def _save_current_net(self, score):
        init_weights = dict()
        for name, network in self.networks.items():
            network.cpu()
            init_weights[name] = copy.deepcopy(network.state_dict())
            self.cuda_func(network)

        self.best_net = (score, init_weights)

    def init_optimizers(self, param):
        optimizers = dict()
        for optimizer_name, optimizer_param in param.items():
            nets_to_optim = list()
            for name in optimizer_param['associated_net']:
                nets_to_optim += self.networks[name].get_training_layers()
            if optimizer_param['optimizer_type'] == "SGD":
                optimizer = optim.SGD(
                    nets_to_optim,
                    **optimizer_param['param']
                )
            elif optimizer_param['optimizer_type'] == "ADAM":
                optimizer = optim.Adam(
                    nets_to_optim,
                    **optimizer_param['param']
                )
            else:
                logger.error('Unknown optimizer type {}'.format(optimizer_param['optimizer_type']))
                raise KeyError('Unknown optimizer type {}'.format(optimizer_param['optimizer_type']))
            optimizers[optimizer_name] = optimizer

        return optimizers

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
        for network in self.networks.values():
            network.cpu()
        for optimizer in self.optimizers.values():
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cpu()

        ser = {
            'best_network': copy.deepcopy(self.best_net),
            'loss': copy.deepcopy(self.loss_log),
            'val_score': copy.deepcopy(self.val_score)
        }

        for name, network in self.networks.items():
            ser['network_' + name] = copy.deepcopy(network.state_dict())
            self.cuda_func(network)
        for name, optimizer in self.optimizers.items():
            ser['optimizer_' + name] = copy.deepcopy(optimizer.state_dict())
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = self.cuda_func(v)
        return ser

    def load(self, datas):
        for network in self.networks.values():
            network.cpu()
        for optimizer in self.optimizers.values():
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cpu()

        self.best_net = datas['best_network']
        self.loss_log = datas['loss']
        self.val_score = datas['val_score']

        self.optimizers = self.init_optimizers(self.optimizers_params)

        for name, network in self.networks.items():
            network.load_state_dict(datas['network_' + name])
            self.cuda_func(network)

        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(datas['optimizer_' + name])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = self.cuda_func(v)
