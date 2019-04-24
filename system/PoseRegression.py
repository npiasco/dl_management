import setlog
import tqdm
import yaml
import os
import system.BaseClass as BaseClass
import torch
import copy
import datasets.SevenScene              # Needed for class creation with eval
import datasets.PoseCambridge
import trainers.loss_functions
import trainers.PoseTrainers
import networks.Pose
import networks.CustomArchi
import networks.ICPNet
import networks.PointNet
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision as torchvis
import pose_utils.utils as pc_utils
import sklearn.preprocessing as skpre
import sklearn.cluster as skclust


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
            if net_part_name == 'self':
                self.trainer.network.load_state_dict(torch.load(os.environ['CNN_WEIGHTS'] + weight_path))
            else:
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

        env_var = 'SEVENSCENES'

        with open(self.root + self.dataset_file, 'rt') as f:
            dataset_params = yaml.safe_load(f)
            logger.debug('dataset param files {} is:'.format(self.root + self.dataset_file))
            logger.debug(yaml.safe_dump(dataset_params))

        self.data = dict()
        self.data['train'] = self.creat_dataset(dataset_params['train'],
                                                os.environ[dataset_params['train'].get('env', env_var)])
        self.data['test'] = dict()
        self.data['test']['queries'] = self.creat_dataset(dataset_params['test']['queries'],
                                                          os.environ[dataset_params['test']['queries'].get('env', env_var)])
        self.data['test']['data'] = self.creat_dataset(dataset_params['test']['data'],
                                                       os.environ[dataset_params['test']['data'].get('env', env_var)])
        self.data['val'] = dict()
        self.data['val']['queries'] = self.creat_dataset(dataset_params['val']['queries'],
                                                         os.environ[dataset_params['val']['queries'].get('env', env_var)])
        self.data['val']['data'] = self.creat_dataset(dataset_params['val']['data'],
                                                      os.environ[dataset_params['val']['data'].get('env', env_var)])

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
                if net_part_name == 'self':
                    self.trainer.networks[name_network].load_state_dict(
                        torch.load(os.environ['CNN_WEIGHTS'] + weight_path)
                    )
                else:
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

            if hasattr(nets_to_test[net_name], 'full_save') is False:
                continue
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

    def threshold_selection(self, final=False, n_values=200, beg=0.5, end=1.0, dataset='val', load=False):
        if not load:
            nets_to_test = self.trainer.networks
            if not final:
                nets_to_test = dict()
                for name, network in self.trainer.networks.items():
                    nets_to_test[name] = copy.deepcopy(network)
                    try:
                        nets_to_test[name].load_state_dict(self.trainer.best_net[1][name])
                    except KeyError:
                        logger.warning("Unable to load best weights for net {}".format(name))

            for network in nets_to_test.values():
                network.eval()

            dtload = data.DataLoader(self.data[dataset]['data'], batch_size=1, shuffle=False, num_workers=8)
            variables = dict()
            logger.info('Db feats computating...')
            for b in tqdm.tqdm(dtload):
                with torch.no_grad():
                    for action in self.trainer.eval_forwards['data']:
                        variables['batch'] =  self.trainer.batch_to_device(b)
                        variables = self.trainer._sequential_forward(action, variables, nets_to_test)

            dtload = data.DataLoader(self.data[dataset]['queries'], batch_size=1, shuffle=False, num_workers=8)

            scores = list()
            pose_err = {'p': list(), 'q': list()}
            refined_pose_err = {'p': list(), 'q': list()}

            for b in tqdm.tqdm(dtload):
                with torch.no_grad():
                    variables['batch'] = self.trainer.batch_to_device(b)

                    for action in self.trainer.eval_forwards['queries']:
                        variables = self.trainer._sequential_forward(action, variables, nets_to_test)

                    gt_pose = trainers.minning_function.recc_acces(variables, ['batch', 'pose'])
                    pose = trainers.minning_function.recc_acces(variables, ['posenet_pose'])
                    refined_pose = trainers.minning_function.recc_acces(variables, ['icp_pose'])
                    scores.append(refined_pose['score'])
                    pose_err['p'].append(torch.norm(gt_pose['p'].squeeze() - pose['p'].squeeze()).item())
                    pose_err['q'].append(2*torch.acos(torch.abs(gt_pose['q'].squeeze().dot(pose['q'].squeeze())))*180/3.14159260)
                    refined_pose_err['p'].append(torch.norm(gt_pose['p'].squeeze() - refined_pose['p'].squeeze()).item())
                    refined_pose_err['q'].append(2 * torch.acos(torch.abs(gt_pose['q'].squeeze().dot(refined_pose['q'].squeeze()))) * 180 / 3.14159260)

            torch.save(scores, 'th_score.pth')
            torch.save(pose_err, 'th_pose.pth')
            torch.save(refined_pose_err, 'th_refined.pth')
        else:
            scores = torch.load('th_score.pth', map_location='cpu')
            pose_err = torch.load('th_pose.pth', map_location='cpu')
            refined_pose_err = torch.load('th_refined.pth', map_location='cpu')

        step = (end - beg)/n_values

        step_list = [s*step for s in range(round(beg/step), round(end/step) + 1)]
        tscores = torch.tensor(scores)
        position_err = torch.tensor(pose_err['p'])*1e2 # cm
        ref_position_err = torch.tensor(refined_pose_err['p'])*1e2 # cm
        med_position_curve = [torch.median(
            torch.cat([position_err[tscores < step_score], ref_position_err[tscores >= step_score]])).item()
                          for step_score in step_list]
        mean_position_curve = [torch.mean(
            torch.cat([position_err[tscores < step_score], ref_position_err[tscores >= step_score]])).item()
                          for step_score in step_list]


        ori_err = torch.tensor(pose_err['q'])
        ref_ori_err = torch.tensor(refined_pose_err['q'])
        med_orientation_curve = [torch.median(
            torch.cat([ori_err[tscores < step_score], ref_ori_err[tscores >= step_score]])).item()
                          for step_score in step_list]
        mean_orientation_curve = [torch.mean(
            torch.cat([ori_err[tscores < step_score], ref_ori_err[tscores >= step_score]])).item()
                          for step_score in step_list]


        fig1 = plt.figure(1)
        plt.plot(step_list, med_position_curve)
        plt.plot(step_list, mean_position_curve)
        plt.legend(['Med','Mean'])
        plt.title('Position (m)')
        fig2 = plt.figure(2)
        plt.plot(step_list, med_orientation_curve)
        plt.plot(step_list, mean_orientation_curve)
        plt.title('Orientation (°)')
        plt.legend(['Med', 'Mean'])
        fig3 = plt.figure(3)
        plt.plot(step_list, med_position_curve)
        plt.plot(step_list, med_orientation_curve)

        plt.show()

    def creat_clusters(self, size_cluster, n_ex=5e5, size_feat=256,
                       jobs=-1, mod='rgb', map_feat='conv7'):
        self.trainer.networks['Main'].train()
        dataset_loader = data.DataLoader(self.data['val']['data'], batch_size=1, num_workers=8, shuffle=True)
        logger.info('Computing feats for clustering')
        feats = list()
        with torch.no_grad():
            for example in tqdm.tqdm(dataset_loader):
                example = self.trainer.batch_to_device(example)
                feat = self.trainer.networks['Main'](self.trainer.cuda_func(example[mod]))[map_feat]
                max_sample = feat.size(2)*feat.size(3)
                feat = feat.view(feat.size(0), size_feat, max_sample).transpose(1, 2).contiguous()
                feat = feat.view(-1, size_feat).cpu().data.numpy()

                feats.append(feat)

        logger.info('Normalizing feats')
        normalized_feats = list()
        for feature in tqdm.tqdm(feats):
            normalized_feats += [f.tolist() for f in feature]
            if len(normalized_feats) >= n_ex:
                break

        normalized_feats = skpre.normalize(normalized_feats)
        logger.info('Computing clusters')
        kmean = skclust.KMeans(n_clusters=size_cluster, n_jobs=jobs)
        kmean.fit(normalized_feats)
        torch_clusters = torch.FloatTensor(kmean.cluster_centers_).unsqueeze(0).transpose(1, 2)

        torch.save(torch_clusters, 'kmean_' + str(size_cluster) + '_clusters_' + map_feat + '.pth')

    def compute_mean_std(self, jobs=16, **kwargs):
        training = kwargs.pop('training', True)
        mod = kwargs.pop('mod', 'rgb')
        testing = kwargs.pop('testing', False)
        val = kwargs.pop('val', True)

        if kwargs:
            raise ValueError('Not expected arg {}'.kwargs)

        mean = None
        std = None
        logger.info('Computing mean and std for modality {}'.format(mod))

        channel = 1 if mod != 'rgb' else 3

        n_ex = 0

        if training:
            n_ex += len(self.data['train'])
            dtload = data.DataLoader(self.data['train'], batch_size=1, num_workers=jobs)
            for batch in tqdm.tqdm(dtload):
                if mean is None:
                    mean = torch.mean(batch[mod].squeeze().view(channel, -1).transpose(0, 1), 0)
                    std = torch.std(batch[mod].squeeze().view(channel, -1).transpose(0, 1), 0)
                else:
                    mean += torch.mean(batch[mod].squeeze().view(channel, -1).transpose(0, 1), 0)
                    std += torch.std(batch[mod].squeeze().view(channel, -1).transpose(0, 1), 0)

        if val:
            n_ex += len(self.data['val']['queries'])
            dtload = data.DataLoader(self.data['val']['queries'], batch_size=1, num_workers=jobs)
            for batch in tqdm.tqdm(dtload):
                if mean is None:
                    mean = torch.mean(batch[mod].squeeze().view(channel, -1).transpose(0, 1), 0)
                    std = torch.std(batch[mod].squeeze().view(channel, -1).transpose(0, 1), 0)
                else:
                    mean += torch.mean(batch[mod].squeeze().view(channel, -1).transpose(0, 1), 0)
                    std += torch.std(batch[mod].squeeze().view(channel, -1).transpose(0, 1), 0)

        if testing:
            n_ex += len(self.data['test']['queries'])
            dtload = data.DataLoader(self.data['test']['queries'], batch_size=1, num_workers=jobs)
            for batch in tqdm.tqdm(dtload):
                if mean is None:
                    mean = torch.mean(batch[mod].squeeze().view(channel, -1).transpose(0, 1), 0)
                    std = torch.std(batch[mod].squeeze().view(channel, -1).transpose(0, 1), 0)
                else:
                    mean += torch.mean(batch[mod].squeeze().view(channel, -1).transpose(0, 1), 0)
                    std += torch.std(batch[mod].squeeze().view(channel, -1).transpose(0, 1), 0)
        mean /= n_ex
        std /= n_ex
        logger.info('Mean = {}\nSTD = {}'.format(mean, std))

    def creat_model(self, fake_depth=False, scene='heads/', test=False, final=False, reduce_fact=2, cuda=True):

        nets_to_test = self.trainer.networks
        if not final:
            nets_to_test = dict()
            for name, network in self.trainer.networks.items():
                nets_to_test[name] = copy.deepcopy(network)
                nets_to_test[name].load_state_dict(self.trainer.best_net[1][name])

        for network in nets_to_test.values():
            network.eval()

        frame_spacing = 20
        size_dataset = len(self.data['test']['queries']) if test else len(self.data['train'])
        sequences = 'TestSplit.txt' if test else 'TrainSplit.txt'
        map_args = {
            'T': torch.eye(4, 4).cuda() if cuda else torch.eye(4, 4),
            'dataset': 'SEVENSCENES',
            'scene': scene,
            'sequences': sequences,
            'num_pc': size_dataset//frame_spacing,
            'resize': 0.1166666667,
            'frame_spacing': frame_spacing,
            'output_size': None,
            'cnn_descriptor': False,
            'cnn_depth': fake_depth,
            'cnn_enc': nets_to_test['Main'].cuda() if cuda else nets_to_test['Main'].cup(),
            'cnn_dec': nets_to_test['Deconv'].cuda() if cuda else nets_to_test['Deconv'].cup(),
            'no_grad': True,
            'reduce_fact': reduce_fact
        }
        file_name = 'fake_depth_model.ply' if fake_depth else 'depth_model.ply'
        file_name = 'test_' + file_name if test else file_name
        color = [255, 0, 0]
        if test:
            color[1] = 255
        if final:
            color[2] = 255
        pc_utils.model_to_ply(map_args=map_args, file_name=file_name, color=color)

    def map_print(self, final=False, mod='rgb', aux_mod='depth', batch_size=1, shuffle=False, save=False, show=True,
                  vmax=3):
        nets_to_test = self.trainer.networks
        if not final:
            nets_to_test = dict()
            for name, network in self.trainer.networks.items():
                nets_to_test[name] = copy.deepcopy(network)
                try:
                    nets_to_test[name].load_state_dict(self.trainer.best_net[1][name])
                except KeyError:
                    logger.warning("Unable to load weights of network {}".format(name))

        for network in nets_to_test.values():
            network.eval()

        dataset = 'test'
        mode = 'queries'
        self.data[dataset][mode].used_mod = [mod, aux_mod]
        torch.manual_seed(0)
        dtload = data.DataLoader(self.data[dataset][mode], batch_size=batch_size, shuffle=shuffle)
        #dtload = data.DataLoader(self.data['train'], batch_size=batch_size, shuffle=True)
        ccmap = plt.get_cmap('jet', lut=1024)

        for i, b in enumerate(dtload):
            with torch.no_grad():
                b = self.trainer.batch_to_device(b)
                _, _, h, w = b[mod].size()
                _, _, haux, waux = b[aux_mod].size()
                main_mod = b[mod].contiguous().view(batch_size, 3, h, w)


                variables = {'batch': b}
                for action in self.trainer.eval_forwards['queries']:
                    variables = self.trainer._sequential_forward(action, variables, nets_to_test)
                output = trainers.minning_function.recc_acces(variables, self.trainer.access_pose[0])
                if isinstance(output, list):
                    plt.figure(0)
                    images_batch = torch.cat((modality,
                                              1 / output[-1] - 1,
                                              torch.nn.functional.interpolate(1 / output[-2] - 1, scale_factor=2,
                                                                              mode='nearest'),
                                              torch.nn.functional.interpolate(1 / output[-3] - 1, scale_factor=4,
                                                                              mode='nearest'),
                                              torch.nn.functional.interpolate(1 / output[-4] - 1, scale_factor=8,
                                                                              mode='nearest'))).cpu().detach()#.clamp(max=modality.max())

                    grid = torchvis.utils.make_grid(images_batch, nrow=batch_size)
                    plt.imshow(grid.numpy().transpose(1, 2, 0)[:, :, 0])

                    output = output[-1]

                if aux_mod in ('rgb'):
                    modality = output
                else:
                    modality = b[aux_mod].contiguous().view(batch_size, -1, haux, waux)

            plt.figure(1)
            images_batch = torch.cat((modality, output)).cpu()
            grid = torchvis.utils.make_grid(images_batch, nrow=batch_size)
            plt.imshow(grid.numpy().transpose(1, 2, 0)[:, :, 0], cmap=ccmap)
            if show:
                plt.imshow(grid.numpy().transpose(1, 2, 0)[:, :, 0], cmap=ccmap, vmin=0, vmax=vmax)
                plt.colorbar()
            if save:
                grid = torchvis.utils.make_grid(modality.cpu(), nrow=batch_size)
                plt.imsave('images/gt_{}.png'.format(i), grid.numpy().transpose(1, 2, 0)[:, :, 0],
                           cmap=ccmap, vmin=0, vmax=vmax)
                grid = torchvis.utils.make_grid(output.cpu(), nrow=batch_size)
                plt.imsave('images/modality_{}.png'.format(i), grid.numpy().transpose(1, 2, 0)[:, :, 0],
                           cmap=ccmap, vmin=0, vmax=vmax)

            plt.figure(2)
            images_batch = torch.cat((torch.abs(modality - output), )).detach().cpu()
            grid = torchvis.utils.make_grid(images_batch, nrow=batch_size)
            if save:
                plt.imsave('images/diff_{}.png'.format(i), grid.numpy().transpose(1, 2, 0), cmap=None)
            if show:
                plt.imshow(grid.numpy().transpose(1, 2, 0), cmap=None)

            plt.figure(3)
            grid = torchvis.utils.make_grid(main_mod.cpu(), nrow=batch_size)
            if save:
                plt.imsave('images/image_{}.png'.format(i), grid.numpy().transpose(1, 2, 0))
            if show:
                plt.imshow(grid.numpy().transpose(1, 2, 0))

            if show:
                plt.show()

    def view_localization(self, final=False, pas=100):
        nets_to_test = self.trainer.networks
        if not final:
            nets_to_test = dict()
            for name, network in self.trainer.networks.items():
                nets_to_test[name] = copy.deepcopy(network)
                try:
                    nets_to_test[name].load_state_dict(self.trainer.best_net[1][name])
                except KeyError:
                    logger.warning("Unable to load best weights for net {}".format(name))

        for network in nets_to_test.values():
            network.eval()

        dataset = 'test'
        mode = 'queries'
        self.data['test']['queries'].used_mod = self.testing_mod
        self.data['test']['data'].used_mod = self.testing_mod

        dtload = data.DataLoader(self.data['test']['data'], batch_size=1, shuffle=False, num_workers=8)
        variables = dict()
        logger.info('Db feats computating...')
        for b in tqdm.tqdm(dtload):
            with torch.no_grad():
                for action in self.trainer.eval_forwards['data']:
                    variables['batch'] =  self.trainer.batch_to_device(b)
                    variables = self.trainer._sequential_forward(action, variables, nets_to_test)

        dtload = data.DataLoader(self.data[dataset][mode], batch_size=1, shuffle=False)

        for b in dtload:
            with torch.no_grad():
                variables['batch'] = self.trainer.batch_to_device(b)
                variables['ref_data'] = self.data['test']['data']

                for action in self.trainer.eval_forwards['queries']:
                    variables = self.trainer._sequential_forward(action, variables, nets_to_test)

            #ref_pc = trainers.minning_function.recc_acces(variables, ['model']).squeeze()
            try:
                ref_pc = trainers.minning_function.recc_acces(variables, ['model', 'pc']).squeeze()
            except (KeyError,TypeError):
                try:
                    ref_pc = trainers.minning_function.recc_acces(variables, ['model',]).squeeze()
                except (KeyError, TypeError):
                    ref_pc = trainers.minning_function.recc_acces(variables, ['ref_pcs', 0]).squeeze()
            #output_pose = trainers.minning_function.recc_acces(variables, ['Tf', 'T'])[0]
            #output_pose = trainers.minning_function.recc_acces(variables, ['icp', 'poses', 'T'])[0]
            output_pose = trainers.minning_function.recc_acces(variables, self.trainer.access_pose + ['T'])[0]
            pc = trainers.minning_function.recc_acces(variables, ['pc']).squeeze()
            output_pc = output_pose.matmul(pc)
            gt_pose = trainers.minning_function.recc_acces(variables, ['batch', 'pose', 'T'])[0]
            gt_pc = gt_pose.matmul(pc)
            if 'posenet_pose' in variables.keys():
                posenet_pose = trainers.minning_function.recc_acces(variables, ['posenet_pose', 'T'])[0]
                posenet_pc = posenet_pose.matmul(pc)

            print('Real pose:')
            print(gt_pose)

            print('Computed pose:')
            print(output_pose)

            if 'posenet_pose' in variables.keys():
                print('Posenet pose:')
                print(posenet_pose)

            print('Diff distance = {} m'.format(torch.norm(gt_pose[:3, 3] - output_pose[:3, 3]).item()))
            gtq = trainers.minning_function.recc_acces(variables, ['batch', 'pose', 'q'])[0]
            #q = trainers.minning_function.recc_acces(variables, ['icp', 'poses', 'q'])[0]
            q = trainers.minning_function.recc_acces(variables, self.trainer.access_pose + ['q'])[0]
            print('Diff orientation = {} deg'.format(2 * torch.acos(torch.abs(gtq.dot(q))) * 180 / 3.14159260))
            if 'posenet_pose' in variables.keys():
                print('Diff distance = {} m (posenet)'.format(torch.norm(gt_pose[:3, 3] - posenet_pose[:3, 3]).item()))
                posenetq = trainers.minning_function.recc_acces(variables, ['posenet_pose', 'q'])[0]
                print('Diff orientation = {} deg (posenet)'.format(2*torch.acos(torch.abs(gtq.dot(posenetq)))*180/3.14159260))
            if 'noised_T' in variables.keys():
                noise_pose = trainers.minning_function.recc_acces(variables, ['noised_T']).squeeze()
                print('Noised pose:')
                print(noise_pose)
                print('Diff distance = {} m (noise T)'.format(torch.norm(gt_pose[:3, 3] - noise_pose[:3, 3]).item()))

            fig = plt.figure(1)
            ax = fig.add_subplot(111, projection='3d')
            plot_size = 60
            pc_utils.plt_pc(ref_pc.cpu(), ax, pas, 'b', size=plot_size)
            pc_utils.plt_pc(gt_pc.cpu(), ax, pas, 'c', size=2*plot_size, marker='*')
            if 'posenet_pose' in variables.keys():
                pc_utils.plt_pc(posenet_pc.cpu(), ax, pas, 'm', size=2*plot_size, marker='o')
                plt.plot([gt_pose[0, 3], posenet_pose[0, 3]],
                         [gt_pose[1, 3], posenet_pose[1, 3]],
                         [gt_pose[2, 3], posenet_pose[2, 3]], color='m')
            plt.plot([gt_pose[0, 3], output_pose[0, 3]],
                     [gt_pose[1, 3], output_pose[1, 3]],
                     [gt_pose[2, 3], output_pose[2, 3]], color='r')
            pc_utils.plt_pc(output_pc.cpu(), ax, pas, 'r', size=2*plot_size, marker='o')
            centroid = torch.mean(ref_pc[:3, :], -1)

            ax.set_xlim([centroid[0].cpu().item()-1, centroid[0].cpu().item()+1])
            ax.set_ylim([centroid[1].cpu().item()-1, centroid[1].cpu().item()+1])
            ax.set_zlim([centroid[2].cpu().item()-1, centroid[2].cpu().item()+1])

            plt.show()

    def test_on_final(self):
        self.data['test']['queries'].used_mod = self.testing_mod
        self.data['test']['data'].used_mod = self.testing_mod
        self.results = self.trainer.test(dataset=self.data['test'],
                                         score_functions=self.test_func,
                                         final=True)
        return self.results


if __name__ == '__main__':
    system = Default(root=os.environ['DATA'] + 'PoseReg/')
    system.train()
    system.test()
    system.plot()
