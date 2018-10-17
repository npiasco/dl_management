import setlog
import yaml
import os
import system.BaseClass as BaseClass
import torch
import copy
import datasets.SevenScene              # Needed for class creation with eval
import trainers.loss_functions
import trainers.PoseTrainers
import networks.Pose
import networks.CustomArchi


logger = setlog.get_logger(__name__)


class Default(BaseClass.Base):
    def __init__(self, **kwargs):
        self.dataset_file = kwargs.pop('dataset_file', 'dataset.yaml')
        BaseClass.Base.__init__(self, **kwargs)

        env_var = os.environ['SEVENSCENES']

        with open(self.root + self.dataset_file, 'rt') as f:
            dataset_params = yaml.safe_load(f)
            logger.debug('dataset param files {} is:'.format(self.root + self.dataset_file))
            logger.debug(yaml.safe_dump(dataset_params))

        self.data = dict()
        self.data['train'] = self.creat_dataset(dataset_params['train'], env_var)
        self.data['test'] = self.creat_dataset(dataset_params['test'], env_var)
        self.data['val'] = self.creat_dataset(dataset_params['val'], env_var)

        net = self.creat_network(self.network_params)

        pos_loss = self.trainer_params['param_class'].pop('pos_loss',
                                                          'trainers.loss_functions.mean_dist')
        ori_loss = self.trainer_params['param_class'].pop('ori_loss',
                                                          'trainers.loss_functions.mean_dist')
        combining_loss = self.trainer_params['param_class'].pop('combining_loss',
                                                                'trainers.loss_functions.AlphaWeights')

        self.trainer_params['param_class'].update(net)
        self.trainer = eval(self.trainer_params['class'])(pos_loss=eval(pos_loss),
                                                          ori_loss=eval(ori_loss),
                                                          combining_loss=eval(combining_loss)(),
                                                          **self.trainer_params['param_class'])
        init_weights = self.network_params.get('init_weights', dict())

        if self.score_file is not None:
            self.load()
        elif self.curr_epoch == 0:
            self.load_initial_net(init_weights)

        if self.curr_epoch != 0:
            self.load()

    def load_initial_net(self, init_weights):
        for net_part_name, weight_path in init_weights.items():
            logger.info(
                'Loading pretrained weights {} (part {})'.format(weight_path, net_part_name))
            getattr(self.trainer.network, net_part_name).load_state_dict(
                torch.load(os.environ['CNN_WEIGHTS'] + weight_path)
            )

    @staticmethod
    def creat_network(network_params):
        return {'network': eval(network_params['class'])(
            **network_params['param_class']
        )}

    def train(self):
        self.data['train'].used_mod = self.training_mod
        self.data['val'].used_mod = self.testing_mod
        BaseClass.Base.train(self)

    def test(self):
        self.data['test'].used_mod = self.testing_mod
        BaseClass.Base.test(self)

    def plot(self, **kwargs):
        BaseClass.Base.plot(self, **kwargs, size_dataset=len(self.data['train']))


class MultNet(Default):
    def __init__(self, **kwargs):
        self.dataset_file = kwargs.pop('dataset_file', 'dataset.yaml')
        BaseClass.Base.__init__(self, **kwargs)

        env_var = os.environ['SEVENSCENES']

        with open(self.root + self.dataset_file, 'rt') as f:
            dataset_params = yaml.safe_load(f)
            logger.debug('dataset param files {} is:'.format(self.root + self.dataset_file))
            logger.debug(yaml.safe_dump(dataset_params))

        self.data = dict()
        self.data['train'] = self.creat_dataset(dataset_params['train'], env_var)
        self.data['test'] = dict()
        self.data['test']['queries'] = self.creat_dataset(dataset_params['test']['queries'], env_var)
        self.data['test']['data'] = self.creat_dataset(dataset_params['test']['data'], env_var)
        self.data['val'] = dict()
        self.data['val']['queries'] = self.creat_dataset(dataset_params['val']['queries'], env_var)
        self.data['val']['data'] = self.creat_dataset(dataset_params['val']['data'], env_var)

        net = self.creat_network(self.network_params)

        self.trainer_params['param_class'].update(net)
        self.trainer = eval(self.trainer_params['class'])(**self.trainer_params['param_class'])
        init_weights = self.network_params.get('init_weights', dict())
        if self.score_file is not None:
            self.load()
        elif self.curr_epoch == 0:
            self.load_initial_net(init_weights)

        if self.curr_epoch != 0:
            self.load()

    def load_initial_net(self, init_weights):
        for name_network, net_part in init_weights.items():
            for net_part_name, weight_path in net_part.items():
                logger.info('Loading pretrained weights {} for network {} (part {})'.format(weight_path, name_network,
                                                                                            net_part_name))
                getattr(self.trainer.networks[name_network], net_part_name).load_state_dict(
                    torch.load(os.environ['CNN_WEIGHTS'] + weight_path)
                )

    @staticmethod
    def creat_network(networks_params):
        existings_networks = {}
        for network_name, network_params in networks_params.items():
            if network_name != 'init_weights':
                existings_networks[network_name] = eval(network_params['class'])(**network_params['param_class'])

        return {'networks': existings_networks}

    def serialize_net(self, final=False, discard_tf=False):
        nets_to_test = dict()
        for net_name, network in self.trainer.networks.items():
            nets_to_test[net_name] = copy.deepcopy(network)
            if not final:
                nets_to_test[net_name].load_state_dict(self.trainer.best_net[1][net_name])

            serlz = nets_to_test[net_name].cpu().full_save(discard_tf=discard_tf)
            for part_name, data in serlz.items():
                serialization_name = net_name + '_' + part_name + '.pth'
                if discard_tf:
                    serialization_name = 'NoJet_' + serialization_name
                torch.save(data, self.root + serialization_name)

    def train(self):
        self.data['train'].used_mod = self.training_mod
        self.data['val']['queries'].used_mod = self.testing_mod
        self.data['val']['data'].used_mod = self.testing_mod
        BaseClass.Base.train(self)

    def test(self):
        self.data['test']['queries'].used_mod = self.testing_mod
        self.data['test']['data'].used_mod = self.testing_mod
        BaseClass.Base.test(self)

if __name__ == '__main__':
    system = Default(root=os.environ['DATA'] + 'PoseReg/')
    system.train()
    system.test()
    system.plot()
