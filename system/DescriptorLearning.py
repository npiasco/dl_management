import setlog
import yaml
import os
import system.BaseClass as BaseClass
import torch.utils.data as data
import torchvision as torchvis
import matplotlib.pyplot as plt
import torch.nn.functional
import torch.autograd as auto
import trainers.minning_function
import trainers.TripletTrainers
import trainers.loss_functions
import datasets.Robotcar                # Needed for class creation with eval
import networks.Descriptor              # Needed for class creation with eval
import networks.Discriminator           # Needed for class creation with eval
import copy
import tqdm
import sklearn.cluster as skclust
import sklearn.preprocessing as skpre


logger = setlog.get_logger(__name__)


class Default(BaseClass.Base):
    def __init__(self, **kwargs):
        self.dataset_file = kwargs.pop('dataset_file', 'dataset.yaml')
        BaseClass.Base.__init__(self, **kwargs)

        env_var = os.environ['ROBOTCAR']

        with open(self.root + self.dataset_file, 'rt') as f:
            dataset_params = yaml.safe_load(f)
            logger.debug('dataset param files {} is:'.format(self.root + self.dataset_file))
            logger.debug(yaml.safe_dump(dataset_params))

        self.data = dict()
        training_param = dict()
        training_param['main'] = self.creat_dataset(dataset_params['train']['param_class']['main'], env_var)
        dataset_params['train']['param_class'].pop('main')
        training_param['examples'] = [self.creat_dataset(d, env_var)
                                      for d in dataset_params['train']['param_class']['examples']]
        dataset_params['train']['param_class'].pop('examples')
        self.data['train'] = eval(dataset_params['train']['class'])(**training_param,
                                                                    **dataset_params['train']['param_class'])

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
        self.data['val']['queries'].used_mod = self.testing_mod
        self.data['val']['data'].used_mod = self.testing_mod
        BaseClass.Base.train(self)

    def test(self):
        self.data['test']['queries'].used_mod = self.testing_mod
        self.data['test']['data'].used_mod = self.testing_mod
        return BaseClass.Base.test(self)

    def plot(self, **kwargs):
        BaseClass.Base.plot(self, **kwargs, size_dataset=len(self.data['train']))

    def print(self, dataset_name):
        if dataset_name == 'train':
            self.data[dataset_name].used_mod = self.training_mod
            dtload = data.DataLoader(self.data[dataset_name], batch_size=4)
        elif dataset_name == 'val_query':
            self.data['val']['queries'].used_mod = self.testing_mod
            dtload = data.DataLoader(self.data['val']['queries'], batch_size=16)
        elif dataset_name == 'val_data':
            self.data['val']['data'].used_mod = self.testing_mod
            dtload = data.DataLoader(self.data['val']['data'], batch_size=16)
        elif dataset_name == 'test_query':
            self.data['test']['queries'].used_mod = self.testing_mod
            dtload = data.DataLoader(self.data['test']['queries'], batch_size=16)
        elif dataset_name == 'test_data':
            self.data['test']['data'].used_mod = self.testing_mod
            dtload = data.DataLoader(self.data['test']['data'], batch_size=16)
        else:
            raise AttributeError('No dataset {}'.format(dataset_name))

        for b in dtload:
            self.batch_print(b, dataset_name)
            plt.show()

    def batch_print(self, batch, data_name):
        if data_name == 'train':
            plt.figure(1)
            plt.title('Query')
            self._format_batch(batch['query'])
            for i in range(2, 2 + len(batch['positives'])):
                plt.figure(i)
                plt.title('Positive ' + str(i))
                self._format_batch(batch['positives'][i-2])
            plt.figure(2 + len(batch['positives']))
            plt.title('Negative')
            self._format_batch(batch['negatives'][0])
        else:
            plt.figure(1)
            self._format_batch(batch)

    @staticmethod
    def _format_batch(batch):
        buffer = tuple()
        for name, mod in batch.items():
            if name not in ('coord',):
                buffer += (mod,)
                print(name)
                print(mod)

        images_batch = torch.cat(buffer, 0)
        grid = torchvis.utils.make_grid(images_batch, nrow=4)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))

    def creat_clusters(self, size_cluster, n_ex=1e6, size_feat=256, jobs=-1):
        # TODO: PCA whitening like this
        self.trainer.network.train()
        dataset_loader = data.DataLoader(self.data['val']['data'], batch_size=1, num_workers=8)
        logger.info('Computing feats for clustering')
        feats = list()
        for example in tqdm.tqdm(dataset_loader):
            feat = self.trainer.network(auto.Variable(self.trainer.cuda_func(example[self.trainer.mod]),
                                              requires_grad=False))['feat']
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

        torch.save(torch_clusters, 'kmean_' + str(size_cluster) + '_clusters.pth')

    def compute_mean_std(self, jobs=16, **kwargs):

        training = kwargs.pop('training', True)
        testing = kwargs.pop('testing', False)
        val = kwargs.pop('val', True)

        if kwargs:
            raise ValueError('Not expected arg {}'.kwargs)

        mean = None
        std = None
        logger.info('Computing mean and std for modality {}'.format(self.trainer.mod))

        channel = 1 if self.trainer.mod != 'rgb' else 3
        n_sample = 0
        if training:
            dtload = data.DataLoader(self.data['train'], batch_size=1, num_workers=jobs)
            n_sample += len(self.data['train'])
            for batch in tqdm.tqdm(dtload):
                if mean is None:
                    mean = torch.mean(batch['query'][self.trainer.mod].squeeze().view(channel, -1).transpose(0, 1), 0)
                    std = torch.std(batch['query'][self.trainer.mod].squeeze().view(channel, -1).transpose(0, 1), 0)
                else:
                    mean += torch.mean(batch['query'][self.trainer.mod].squeeze().view(channel, -1).transpose(0, 1), 0)
                    std += torch.std(batch['query'][self.trainer.mod].squeeze().view(channel, -1).transpose(0, 1), 0)
                n_sample += len(batch['positives']) + len(batch['negatives'])
                for ex in batch['positives']:
                    mean += torch.mean(ex[self.trainer.mod].squeeze().view(channel, -1).transpose(0, 1), 0)
                    std += torch.std(ex[self.trainer.mod].squeeze().view(channel, -1).transpose(0, 1), 0)
                for ex in batch['negatives']:
                    mean += torch.mean(ex[self.trainer.mod].squeeze().view(channel, -1).transpose(0, 1), 0)
                    std += torch.std(ex[self.trainer.mod].squeeze().view(channel, -1).transpose(0, 1), 0)
        if val:
            dtload = data.DataLoader(self.data['val']['queries'], batch_size=1, num_workers=jobs)
            n_sample += len(self.data['val']['queries']) + len(self.data['val']['data'])
            for batch in tqdm.tqdm(dtload):
                if mean is None:
                    mean = torch.mean(batch[self.trainer.mod].squeeze().view(channel, -1).transpose(0, 1), 0)
                    std = torch.std(batch[self.trainer.mod].squeeze().view(channel, -1).transpose(0, 1), 0)
                else:
                    mean += torch.mean(batch[self.trainer.mod].squeeze().view(channel, -1).transpose(0, 1), 0)
                    std += torch.std(batch[self.trainer.mod].squeeze().view(channel, -1).transpose(0, 1), 0)
            dtload = data.DataLoader(self.data['val']['data'], batch_size=1, num_workers=jobs)
            for batch in tqdm.tqdm(dtload):
                if mean is None:
                    mean = torch.mean(batch[self.trainer.mod].squeeze().view(channel, -1).transpose(0, 1), 0)
                    std = torch.std(batch[self.trainer.mod].squeeze().view(channel, -1).transpose(0, 1), 0)
                else:
                    mean += torch.mean(batch[self.trainer.mod].squeeze().view(channel, -1).transpose(0, 1), 0)
                    std += torch.std(batch[self.trainer.mod].squeeze().view(channel, -1).transpose(0, 1), 0)
        if testing:
            dtload = data.DataLoader(self.data['test']['queries'], batch_size=1, num_workers=jobs)
            n_sample += len(self.data['test']['queries']) + len(self.data['test']['data'])
            for batch in tqdm.tqdm(dtload):
                if mean is None:
                    mean = torch.mean(batch[self.trainer.mod].squeeze().view(channel, -1).transpose(0, 1), 0)
                    std = torch.std(batch[self.trainer.mod].squeeze().view(channel, -1).transpose(0, 1), 0)
                else:
                    mean += torch.mean(batch[self.trainer.mod].squeeze().view(channel, -1).transpose(0, 1), 0)
                    std += torch.std(batch[self.trainer.mod].squeeze().view(channel, -1).transpose(0, 1), 0)
            dtload = data.DataLoader(self.data['test']['data'], batch_size=1, num_workers=jobs)
            for batch in tqdm.tqdm(dtload):
                if mean is None:
                    mean = torch.mean(batch[self.trainer.mod].squeeze().view(channel, -1).transpose(0, 1), 0)
                    std = torch.std(batch[self.trainer.mod].squeeze().view(channel, -1).transpose(0, 1), 0)
                else:
                    mean += torch.mean(batch[self.trainer.mod].squeeze().view(channel, -1).transpose(0, 1), 0)
                    std += torch.std(batch[self.trainer.mod].squeeze().view(channel, -1).transpose(0, 1), 0)

        logger.info('Mean = {}\nSTD = {}'.format(mean/n_sample, std/n_sample))

    def map_print(self):
        tmp_net = copy.deepcopy(self.trainer.network)
        tmp_net.load_state_dict(self.trainer.best_net[1])
        self.data['train'].used_mod = self.training_mod
        dtload = data.DataLoader(self.data['train'], batch_size=4)
        plt.figure(1)
        plt.figure(2)
        ccmap = plt.get_cmap('jet', lut=1024)

        for b in dtload:
            main_mod = b['query'][self.trainer.mod].contiguous().view(4, -1, 224, 224)

            for name, lay in tmp_net.feature.feature.named_children():
                if name == 'jet_tf':
                    output = lay(
                        auto.Variable(
                            self.trainer.cuda_func(
                                b['query'][self.trainer.mod]
                            ),
                            requires_grad=False
                        )
                    )
                    print(lay.embedding.weight)

            image_batch = output.data.cpu()
            grid = torchvis.utils.make_grid(image_batch, nrow=4)
            plt.figure(1)
            if image_batch.size(1) == 1:
                plt.imshow(grid.numpy().transpose(1, 2, 0)[:, :, 0], cmap=ccmap)
            else:
                plt.imshow(grid.numpy().transpose(1, 2, 0))
            plt.colorbar()
            plt.figure(2)
            grid = torchvis.utils.make_grid(main_mod.cpu(), nrow=4)
            plt.imshow(grid.numpy().transpose(1, 2, 0))
            plt.show()


class Deconv(Default):
    def __init__(self, **kwargs):
        Default.__init__(self, **kwargs)

    def map_print(self, final=False):
        tmp_net = copy.deepcopy(self.trainer.network)
        tmp_net.train()  # To have the infered map
        if not final:
            tmp_net.load_state_dict(self.trainer.best_net[1])
        self.data['train'].used_mod = self.training_mod
        dtload = data.DataLoader(self.data['train'], batch_size=4)
        plt.figure(1)
        plt.figure(2)
        ccmap = plt.get_cmap('jet', lut=1024)

        for b in dtload:
            main_mod = b['query'][self.trainer.mod].contiguous().view(4, 3, 224, 224)
            modality = b['query'][self.trainer.aux_mod].contiguous().view(4, -1, 224, 224)
            output = tmp_net(
                auto.Variable(
                    self.trainer.cuda_func(
                        b['query'][self.trainer.mod]
                    ),
                    requires_grad=False
                )
            )
            print(output['desc'])
            images_batch = torch.cat((modality.cpu(), output['maps'].data.cpu()))
            grid = torchvis.utils.make_grid(images_batch, nrow=4)
            plt.figure(1)
            if images_batch.size(1) == 1:
                plt.imshow(grid.numpy().transpose(1, 2, 0)[:, :, 0], cmap=ccmap)
            else:
                plt.imshow(grid.numpy().transpose(1, 2, 0))
            plt.colorbar()
            plt.figure(2)
            grid = torchvis.utils.make_grid(main_mod.cpu(), nrow=4)
            plt.imshow(grid.numpy().transpose(1, 2, 0))
            plt.show()

    def creat_clusters(self, size_cluster, n_ex=1e6, size_feat=256, jobs=-1, feat_type='main', norm=True):
        self.trainer.network.train()
        dataset_loader = data.DataLoader(self.data['val']['data'], batch_size=1, num_workers=8)
        logger.info('Computing feats for clustering')
        feats = list()
        for example in tqdm.tqdm(dataset_loader):
            feat = self.trainer.network(auto.Variable(self.trainer.cuda_func(example[self.trainer.mod]),
                                              requires_grad=False))['feat'][feat_type]
            max_sample = feat.size(2)*feat.size(3)
            feat = feat.view(feat.size(0), size_feat, max_sample).transpose(1, 2).contiguous()
            feat = feat.view(-1, size_feat).cpu().data.numpy()

            feats.append(feat)

        normalized_feats = list()
        for feature in tqdm.tqdm(feats):
            normalized_feats += [f.tolist() for f in feature]
            if len(normalized_feats) >= n_ex:
                break
        if norm:
            logger.info('Normalizing feats')
            normalized_feats = skpre.normalize(normalized_feats)
        logger.info('Computing clusters')
        kmean = skclust.KMeans(n_clusters=size_cluster, n_jobs=jobs)
        kmean.fit(normalized_feats)
        torch_clusters = torch.Tensor(kmean.cluster_centers_).unsqueeze(0).transpose(1,2)

        torch.save(torch_clusters, 'kmean_' + str(size_cluster) + '_{}_'.format(feat_type) + 'clusters.pth')

class MultNet(Default):
    def __init__(self, **kwargs):
        Default.__init__(self, **kwargs)

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

    def map_print(self, net_name, final=False, mod='rgb', aux_mod='mono_depth', batch_size=4):

        nets_to_test = self.trainer.networks
        if not final:
            nets_to_test = dict()
            for name, network in self.trainer.networks.items():
                nets_to_test[name] = copy.deepcopy(network)
                nets_to_test[name].load_state_dict(self.trainer.best_net[1][name])

        for network in nets_to_test.values():
            network.eval()

        #self.data['train'].used_mod = self.training_mod
        self.data['test']['queries'].used_mod = ['rgb', 'mono_ref', 'mono_depth']
        #dtload = data.DataLoader(self.data['train'], batch_size=batch_size)
        dtload = data.DataLoader(self.data['test']['queries'], batch_size=batch_size)
        plt.figure(1)
        plt.figure(2)
        ccmap = plt.get_cmap('jet', lut=1024)

        for b in dtload:
            main_mod = b[mod].contiguous().view(batch_size, 3, 224, 224)
            true_img = b['mono_ref'].contiguous().view(batch_size, 3, 224, 224)
            modality = b[aux_mod].contiguous().view(batch_size, -1, 224, 224)
            #main_mod = b[mod].contiguous().view(batch_size, 3, 224, 224)
            #modality = b[aux_mod].contiguous().view(batch_size, -1, 224, 224)

            variables = {'batch': b}
            for action in self.trainer.eval_forwards['dataset']:
                variables = self.trainer._sequential_forward(action, variables, nets_to_test)
            output = trainers.minning_function.recc_acces(variables, ['maps'])

            '''
            pruned_maps = trainers.minning_function.random_prunning({'maps': output['maps'], 'gt': auto.Variable(modality)},
                                                                    multiples_instance=False, target=['maps'], mask=['gt'])
            pruned_mod = trainers.minning_function.random_prunning(copy.deepcopy(modality),
                                                                    multiples_instance=False, target=[], prob=0.5)
            images_batch = torch.cat((modality.cpu(), pruned_maps.data.cpu(), output['maps'].data.cpu()))
            '''

            diff_map = torch.abs(modality.cpu()-output.data.cpu())
            #images_batch = torch.cat((modality.cpu(), output.data.cpu(), diff_map))
            images_batch = torch.cat((modality.cpu(), output.data.cpu()))

            grid = torchvis.utils.make_grid(images_batch, nrow=batch_size)
            plt.figure(1)
            if images_batch.size(1) == 1:
                plt.imshow(grid.numpy().transpose(1, 2, 0)[:, :, 0], cmap=ccmap)
            else:
                plt.imshow(grid.numpy().transpose(1, 2, 0))
            #plt.colorbar()
            plt.figure(2)
            #grid = torchvis.utils.make_grid(main_mod.cpu(), nrow=batch_size)
            grid = torchvis.utils.make_grid(true_img.cpu(), nrow=batch_size)
            plt.imshow(grid.numpy().transpose(1, 2, 0))
            """
            plt.figure(3)
            grid = torchvis.utils.make_grid(diff_map, nrow=4)
            plt.imshow(grid.numpy().transpose(1, 2, 0)[:, :, 0], cmap=ccmap)
            plt.colorbar()
            """

            plt.show()

    def creat_clusters(self, size_cluster, n_ex=1e6, size_feat=256, jobs=-1, mod='rgb'):
        # TODO: PCA whitening like this
        self.trainer.networks['Main'].train()
        dataset_loader = data.DataLoader(self.data['val']['data'], batch_size=1, num_workers=8)
        logger.info('Computing feats for clustering')
        feats = list()
        for example in tqdm.tqdm(dataset_loader):
            feat = self.trainer.networks['Main'](auto.Variable(self.trainer.cuda_func(example[mod]),
                                              requires_grad=False))['feat']
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

        torch.save(torch_clusters, 'kmean_' + str(size_cluster) + '_clusters.pth')

    def creat_clusters_fusevlad(self, size_cluster, n_ex=1e6, size_feat=512, jobs=-1):
        nets_to_test = self.trainer.networks
        for network in nets_to_test.values():
            network.eval()

        dataset_loader = data.DataLoader(self.data['val']['data'], batch_size=1, num_workers=8)
        logger.info('Computing feats for clustering')
        feats = list()
        for example in tqdm.tqdm(dataset_loader):
            variables = {'batch': example}

            for action in self.trainer.eval_forwards['dataset']:
                variables = self.trainer._sequential_forward(action, variables, nets_to_test)

            feat = torch.cat((variables['main_feat']['feat'], variables['aux_feat']['feat']), dim=1)

            max_sample = feat.size(2) * feat.size(3)
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

        torch.save(torch_clusters, 'kmeans_' + str(size_cluster) + '_clusters.pth')

    def creat_clusters_auxlad(self, size_cluster, n_ex=1e6, size_feat=512, jobs=-1):
        nets_to_test = self.trainer.networks
        for network in nets_to_test.values():
            network.eval()

        dataset_loader = data.DataLoader(self.data['val']['data'], batch_size=1, num_workers=8)
        logger.info('Computing feats for clustering')
        feats = list()
        for example in tqdm.tqdm(dataset_loader):
            variables = {'batch': example}

            for action in self.trainer.eval_forwards['dataset']:
                variables = self.trainer._sequential_forward(action, variables, nets_to_test)

            feat = variables['aux_out']['feat']

            max_sample = feat.size(2) * feat.size(3)
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

        torch.save(torch_clusters, 'aux_kmeans_' + str(size_cluster) + '_clusters.pth')


if __name__ == '__main__':

    system = Default(root=os.environ['DATA'] + 'DescLearning/Template/')
    system.compute_mean_std()
    # system.creat_clusters(64)
    '''
    system.train()
    system.test()
    system.plot()
    system.print('val_data')
    '''
    '''
    system = Deconv(root=os.environ['DATA'] + 'DescLearning/DeconvTemplate/')
    system.map_print()
    '''
