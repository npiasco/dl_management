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
import networks.ICPNet
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision as torchvis
import pose_utils.utils as pc_utils


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

    def map_print(self, final=False, mod='rgb', aux_mod='depth', batch_size=1):
        nets_to_test = self.trainer.networks
        if not final:
            nets_to_test = dict()
            for name, network in self.trainer.networks.items():
                nets_to_test[name] = copy.deepcopy(network)
                nets_to_test[name].load_state_dict(self.trainer.best_net[1][name])

        for network in nets_to_test.values():
            network.eval()

        dataset = 'test'
        mode = 'queries'
        self.data[dataset][mode].used_mod = [mod, aux_mod]

        dtload = data.DataLoader(self.data[dataset][mode], batch_size=batch_size, shuffle=True)
        ccmap = plt.get_cmap('jet', lut=1024)

        for b in dtload:
            with torch.no_grad():
                b = self.trainer.batch_to_device(b)
                _, _, h, w = b[mod].size()
                _, _, haux, waux = b[aux_mod].size()
                main_mod = b[mod].contiguous().view(batch_size, 3, h, w)
                modality = b[aux_mod].contiguous().view(batch_size, -1, haux, waux)

                variables = {'batch': b}
                for action in self.trainer.eval_forwards['queries']:
                    variables = self.trainer._sequential_forward(action, variables, nets_to_test)
                output = trainers.minning_function.recc_acces(variables, ['maps'])

                inv_output = 1/output - 1
                mean = torch.mean(inv_output.view(inv_output.size(0), -1), 1)
                for nb, im in enumerate(inv_output):
                    inv_output[nb, im>mean[nb]*2] = 0

                inv_mod = 1/(modality + 1)
                if 'filters' in  variables.keys():
                    filted_im = output.new_zeros(output.size())
                    for nb, im in enumerate(trainers.minning_function.recc_acces(variables, ['fw_main', 'conv1'])):
                        ch, h, w = im.size()
                        filted_im[nb] = nets_to_test['Filter'](im.view(ch, -1).t()).t().view(1, h, w)

            plt.figure(1)
            images_batch = torch.cat((modality.cpu(), inv_mod.cpu(), inv_output.detach().cpu(), output.detach().cpu()))
            grid = torchvis.utils.make_grid(images_batch, nrow=batch_size*2)
            plt.imshow(grid.numpy().transpose(1, 2, 0)[:, :, 0], cmap=ccmap)
            plt.colorbar()

            plt.figure(3)
            grid = torchvis.utils.make_grid(main_mod.cpu(), nrow=batch_size)
            plt.imshow(grid.numpy().transpose(1, 2, 0))

            if 'filters' in variables.keys():
                plt.figure(4)
                inv_output = inv_output*filted_im
                images_batch = torch.cat((filted_im.cpu(), inv_output.cpu()))
                grid = torchvis.utils.make_grid(images_batch, nrow=batch_size)
                plt.imshow(grid.numpy().transpose(1, 2, 0)[:, :, 0], cmap=ccmap)
                plt.colorbar()

            plt.show()

    def view_localization(self, final=False, pas=100):
        nets_to_test = self.trainer.networks
        if not final:
            nets_to_test = dict()
            for name, network in self.trainer.networks.items():
                nets_to_test[name] = copy.deepcopy(network)
                nets_to_test[name].load_state_dict(self.trainer.best_net[1][name])

        for network in nets_to_test.values():
            network.eval()

        dataset = 'test'
        mode = 'queries'

        dtload = data.DataLoader(self.data[dataset][mode], batch_size=1)

        for b in dtload:
            with torch.no_grad():
                b = self.trainer.batch_to_device(b)
                variables = {'batch': b}

                for action in self.trainer.eval_forwards['queries']:
                    variables = self.trainer._sequential_forward(action, variables, nets_to_test)

            ref_pc = trainers.minning_function.recc_acces(variables, ['model']).squeeze()
            output_pose = trainers.minning_function.recc_acces(variables, ['Tf', 'T'])[0]
            #output_pose = trainers.minning_function.recc_acces(variables, ['icp', 'poses', 'T'])[0]
            pc = trainers.minning_function.recc_acces(variables, ['pc']).squeeze()
            output_pc = output_pose.matmul(pc)
            gt_pose = trainers.minning_function.recc_acces(variables, ['batch', 'pose', 'T'])[0]
            gt_pc = gt_pose.matmul(pc)
            if 'posenet_pose' in variables.keys():
                #posenet_pose = trainers.minning_function.recc_acces(variables, ['posenet_pose', 'T'])[0]
                posenet_pose = trainers.minning_function.recc_acces(variables, ['posenet_pose', 'T'])[0]
                posenet_pc = posenet_pose.matmul(pc)

            print('Real pose:')
            print(gt_pose)

            print('Computed pose:')
            print(output_pose)

            if 'posenet_pose' in variables.keys():
                print('Posenet pose:')
                print(posenet_pose)

            print('Diff distance = {} m'.format(torch.norm(gt_pose[:, 3] - output_pose[:, 3]).item()))
            if 'posenet_pose' in variables.keys():
                print('Diff distance = {} m (posenet)'.format(torch.norm(gt_pose[:, 3] - posenet_pose[:, 3]).item()))
            if 'noised_T' in variables.keys():
                noise_pose = trainers.minning_function.recc_acces(variables, ['noised_T']).squeeze()
                print('Noised pose:')
                print(noise_pose)
                print('Diff distance = {} m (noise T)'.format(torch.norm(gt_pose[:, 3] - noise_pose[:, 3]).item()))

            fig = plt.figure(1)
            ax = fig.add_subplot(111, projection='3d')

            pc_utils.plt_pc(ref_pc.cpu(), ax, pas, 'b')
            #pc_utils.plt_pc(gt_pc.cpu(), ax, pas, 'c')
            pc_utils.plt_pc(output_pc.cpu(), ax, pas, 'r')
            plt.show()

    def test_on_final(self):
        self.results = self.trainer.test(dataset=self.data['test'],
                                         score_functions=self.test_func,
                                         final=True)
        return self.results


if __name__ == '__main__':
    system = Default(root=os.environ['DATA'] + 'PoseReg/')
    system.train()
    system.test()
    system.plot()
