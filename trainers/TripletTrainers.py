import setlog
import trainers.Base as Base
import torch.nn.functional as func
import torch.autograd as auto
import trainers.minning_function as minning
from trainers.minning_function import recc_acces
import trainers.loss_functions as loss_func
import datasets.Robotcar as Robotcar
import torch.utils as utils
import torch.utils.data
import os
import datasets.multmodtf as tf
import networks.Descriptor as Desc
import numpy as np
import tqdm
import copy
import score.Functions as ScoreFunc


logger = setlog.get_logger(__name__)


class Trainer(Base.BaseTrainer):
    def __init__(self, **kwargs):
        Base.BaseTrainer.__init__(
            self,
            lr=kwargs.pop('lr', 0.0001),
            momentum=kwargs.pop('momentum', 0.9),
            weight_decay=kwargs.pop('weight_decay', 0.001),
            cuda_on=kwargs.pop('cuda_on', True),
            optimizer_type=kwargs.pop('optimizer_type', 'SGD'),
            network=kwargs.pop('network', None),
            val_num_workers=kwargs.pop('val_num_workers', 8)
        )

        self.triplet_loss = kwargs.pop('triplet_loss', {'func': func.triplet_margin_loss,
                                                        'param': dict()})
        self.minning_func = kwargs.pop('minning_func', {'func': minning.hard_minning,
                                                        'param': dict()})
        self.mod = kwargs.pop('mod', 'rgb')
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if isinstance(self.triplet_loss['func'], str):
            self.triplet_loss['func'] = eval(self.triplet_loss['func'])
        if isinstance(self.minning_func['func'], str):
            self.minning_func['func'] = eval(self.minning_func['func'])
        self.optimizer = self.init_optimizer(self.network.get_training_layers())
        self.loss_log['triplet_loss'] = list()

    def train(self, batch):
        self.network.train()
        # Reset gradients
        self.optimizer.zero_grad()
        # dataset.associated_net = copy.deepcopy(net).cpu()

        # Forward pass
        anchor = self.network(auto.Variable(self.cuda_func(batch['query'][self.mod]), requires_grad=True))
        positive = self.minning_func['func'](self.network,
                                             batch,
                                             'positives',
                                             cuda_func=self.cuda_func,
                                             mod=self.mod,
                                             **self.minning_func['param'])
        negative = self.minning_func['func'](self.network,
                                             batch,
                                             'negatives',
                                             cuda_func=self.cuda_func,
                                             mod=self.mod,
                                             **self.minning_func['param'])

        loss = self.triplet_loss['func'](anchor['desc'],
                                         positive['desc'],
                                         negative['desc'],
                                         **self.triplet_loss['param'])

        loss.backward()  # calculate the gradients (backpropagation)
        self.optimizer.step()  # update the weights
        self.loss_log['triplet_loss'].append(loss.data[0])
        logger.debug('Triplet loss is {}'.format(loss.data[0]))

    def eval(self, **kwargs):
        dataset = kwargs.pop('dataset', None)

        score_function = kwargs.pop('score_function', None)
        ep = kwargs.pop('ep', None)
        if kwargs:
            logger.error('Unexpected **kwargs: %r' % kwargs)
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if len(self.val_score) <= ep:
            ranked = self._compute_sim(self.network, dataset['queries'], dataset['data'])
            score = score_function(ranked)
            self.val_score.append(score)
            if score_function.rank_score(score, self.best_net[0]):
                self.network.cpu()
                self.best_net = (score, copy.deepcopy(self.network.state_dict()))
                self.cuda_func(self.network)
        logger.info('Score is: {}'.format(self.val_score[ep]))

    def test(self, **kwargs):
        dataset = kwargs.pop('dataset', None)
        score_functions = kwargs.pop('score_functions', None)
        if kwargs:
            logger.error('Unexpected **kwargs: %r' % kwargs)
            raise TypeError('Unexpected **kwargs: %r' % kwargs)
        net_to_test = copy.deepcopy(self.network)
        net_to_test.load_state_dict(self.best_net[1])
        ranked = self._compute_sim(net_to_test, dataset['queries'], dataset['data'])
        results = dict()
        for function_name, score_func in score_functions.items():
            results[function_name] = score_func(ranked)
        return results

    def _compute_sim(self, network, queries, dataset):
        dataset.used_mod = [self.mod]
        queries.used_mod = [self.mod]
        dataset_loader = utils.data.DataLoader(dataset, batch_size=1, num_workers=self.val_num_workers)
        queries_loader = utils.data.DataLoader(queries, batch_size=1, num_workers=self.val_num_workers)

        network.eval()

        logger.info('Computing dataset feats')
        dataset_feats = [(network(auto.Variable(self.cuda_func(example[self.mod]),
                                                requires_grad=False))[0].cpu().data.numpy(),
                          example['coord'].cpu().numpy()) for example in tqdm.tqdm(dataset_loader)]

        logger.info('Computing similarity')
        ranked = list()
        for query in tqdm.tqdm(queries_loader):
            feat = network(auto.Variable(self.cuda_func(query[self.mod]),
                                         requires_grad=False))[0].cpu().data.numpy()
            gt_pos = query['coord'].cpu().numpy()
            diff = [(np.dot(feat, d_feat[0]), np.linalg.norm(gt_pos - d_feat[1])) for d_feat in dataset_feats]
            sorted_index = list(np.argsort([d[0] for d in diff]))
            ranked.append([diff[idx][1] for idx in reversed(sorted_index)])

        return ranked


class DeconvTrainer(Trainer):
    def __init__(self, **kwargs):
        self.modal_loss = kwargs.pop('modal_loss', {'func': loss_func.l1_modal_loss,
                                                    'param': {'p': 1, 'factor': 1}})
        aux_loss = kwargs.pop('aux_loss', dict())
        self.aux_mod = kwargs.pop('aux_mod', 'depth')

        Trainer.__init__(self, **kwargs)

        if isinstance(self.modal_loss['func'], str):
            self.modal_loss['func'] = eval(self.modal_loss['func'])
        self.loss_log['modal_loss'] = list()

        self.aux_loss = dict()
        for name, info in aux_loss.items():
            self.aux_loss[name] = info
            self.aux_loss[name]['func'] = eval(aux_loss[name]['func'])
            self.loss_log[name] = list()

    def train(self, batch):
        self.network.train()
        # Reset gradients
        self.optimizer.zero_grad()

        # Forward pass
        anchor = self.network(auto.Variable(self.cuda_func(batch['query'][self.mod]), requires_grad=True))
        positive, pos_idx = self.minning_func['func'](self.network,
                                                      batch,
                                                      'positives',
                                                      cuda_func=self.cuda_func,
                                                      mod=self.mod,
                                                      return_idx=True,
                                                      **self.minning_func['param'])
        negative, neg_idx = self.minning_func['func'](self.network,
                                                      batch,
                                                      'negatives',
                                                      mod=self.mod,
                                                      return_idx=True,
                                                      **self.minning_func['param'])

        triplet_loss = self.triplet_loss['func'](anchor['desc'],
                                                 positive['desc'],
                                                 negative['desc'],
                                                 **self.triplet_loss['param'])

        anchor_mod = auto.Variable(self.cuda_func(batch['query'][self.aux_mod]), requires_grad=False)
        pos_mod = [
            auto.Variable(
                self.cuda_func(
                    torch.stack(
                        [batch['positives'][sub_b][self.aux_mod][i] for i, sub_b in enumerate(idxs)],
                        dim=0
                    )
                ),
                requires_grad=False
            ) for idxs in pos_idx
            ]
        neg_mod = [
            auto.Variable(
                self.cuda_func(
                    torch.stack(
                        [batch['negatives'][sub_b][self.aux_mod][i] for i, sub_b in enumerate(idxs)],
                        dim=0
                    )
                ),
                requires_grad=False
            ) for idxs in neg_idx
            ]
        modal_loss = self.modal_loss['func']((anchor['maps'], *positive['maps'], *negative['maps']),
                                             (anchor_mod, *pos_mod, *neg_mod),
                                             **self.modal_loss['param'])

        loss = triplet_loss + modal_loss
        for name, aux_los in self.aux_loss.items():
            val = aux_los['func'](anchor['desc'], positive['desc'], negative['desc'], **aux_los['param'])
            loss += val
            self.loss_log[name].append(val.data[0])
            logger.debug(name + ' loss is {}'.format(val.data[0]))

        loss.backward()  # calculate the gradients (backpropagation)
        self.optimizer.step()  # update the weights
        self.loss_log['triplet_loss'].append(triplet_loss.data[0])
        logger.debug('Triplet loss is {}'.format(triplet_loss.data[0]))
        self.loss_log['modal_loss'].append(modal_loss.data[0])
        logger.debug('Modal loss is {}'.format(modal_loss.data[0]))


class MultNetTrainer(Base.BaseMultNetTrainer):
    def __init__(self, **kwargs):
        training_pipeline = kwargs.pop('training_pipeline', list())
        eval_forwards = kwargs.pop('eval_forwards', dict())
        self.eval_final_desc = kwargs.pop('eval_final_desc', ['desc'])

        Base.BaseMultNetTrainer.__init__(self, **kwargs)

        self.training_pipeline = list()
        for action in training_pipeline:
            self.training_pipeline.append(action)
            if 'func' in self.training_pipeline[-1].keys():
                self.training_pipeline[-1]['func'] = eval(action['func'])
            if self.training_pipeline[-1]['mode'] == 'loss':
                self.loss_log[action['name']] = list()

        self.eval_forwards = {'dataset': list(), 'queries': list()}
        for forward in eval_forwards['dataset']:
            self.eval_forwards['dataset'].append(forward)
            self.eval_forwards['dataset'][-1]['func'] = eval(forward['func'])
        for forward in eval_forwards['queries']:
            self.eval_forwards['queries'].append(forward)
            self.eval_forwards['queries'][-1]['func'] = eval(forward['func'])

    def train(self, batch):
        for network in self.networks.values():
            network.train()
            for params in network.get_training_layers():
                for param in params['params']:
                    param.requires_grad = True

        # Forward pass
        variables = {'batch': batch}
        sumed_loss = 0
        for action in self.training_pipeline:
            if action['mode'] == 'batch_forward':
                variables[action['out_name']] = action['func'](
                    self.networks[action['net_name']],
                    variables,
                    cuda_func=self.cuda_func,
                    **action['param']
                )
            elif action['mode'] == 'forward':
                variables[action['out_name']] = action['func'](
                    self.networks[action['net_name']],
                    variables,
                    **action['param']
                )
            elif action['mode'] == 'minning':
                variables[action['out_name']] = action['func'](
                    variables,
                    **action['param']
                )
            elif action['mode'] == 'loss':
                input_args = [recc_acces(variables, name) for name in action['args']]
                val = action['func'](*input_args, **action['param'])
                sumed_loss += val
                self.loss_log[action['name']].append(val.data[0])
                logger.debug(action['name'] + ' loss is {}'.format(val.data[0]))
            elif action['mode'] == 'backprop':
                self.optimizers[action['trainer']].zero_grad()
                sumed_loss.backward()
                # sumed_loss.backward(retain_graph=True)
                self.optimizers[action['trainer']].step()
                sumed_loss = 0
                for name in self.optimizers_params[action['trainer']]['associated_net']:
                    for params in self.networks[name].get_training_layers():
                        for param in params['params']:
                            param.requires_grad = False
            """
            if 'adver_true_query' in variables.keys() and not variables['adver_true_query']._backward_hooks:
                variables['adver_true_query'].register_hook(lambda x: print('adver_true_query', x))
            if 'adver_false_query' in variables.keys() and not variables['adver_false_query']._backward_hooks:
                variables['adver_false_query'].register_hook(lambda x: print('adver_false_query', x))
            """


    def eval(self, **kwargs):
        dataset = kwargs.pop('dataset', None)

        score_function = kwargs.pop('score_function', None)
        ep = kwargs.pop('ep', None)
        if kwargs:
            logger.error('Unexpected **kwargs: %r' % kwargs)
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if len(self.val_score) <= ep:
            if isinstance(score_function, ScoreFunc.Reconstruction_Error):
                ranked = self._compute_rerror(self.networks, dataset['queries'], dataset['data'])
            else:
                ranked = self._compute_sim(self.networks, dataset['queries'], dataset['data'])

            score = score_function(ranked)
            self.val_score.append(score)
            if score_function.rank_score(score, self.best_net[0]):
                self._save_current_net(score)

        logger.info('Score is: {}'.format(self.val_score[ep]))

    def test(self, **kwargs):
        dataset = kwargs.pop('dataset', None)
        score_functions = kwargs.pop('score_functions', None)
        if kwargs:
            logger.error('Unexpected **kwargs: %r' % kwargs)
            raise TypeError('Unexpected **kwargs: %r' % kwargs)
        nets_to_test = dict()
        for name, network in self.networks.items():
            nets_to_test[name] = copy.deepcopy(network)
            nets_to_test[name].load_state_dict(self.best_net[1][name])

        if True in [isinstance(score_function, ScoreFunc.Reconstruction_Error)
                    for score_function in score_functions.values()]:
            ranked = self._compute_rerror(nets_to_test, dataset['queries'], dataset['data'])
        else:
            ranked = self._compute_sim(nets_to_test, dataset['queries'], dataset['data'])
        results = dict()
        for function_name, score_func in score_functions.items():
            results[function_name] = score_func(ranked)
        return results

    def _compute_rerror(self, networks, queries, dataset):
        dataset_loader = utils.data.DataLoader(dataset, batch_size=1, num_workers=self.val_num_workers)
        queries_loader = utils.data.DataLoader(queries, batch_size=1, num_workers=self.val_num_workers)

        for network in networks.values():
            network.train()

        errors = list()
        # Forward pass
        logger.info('Computing dataset/queries reconstruction error')
        for dataloader in (dataset_loader, queries_loader):
            for batch in tqdm.tqdm(dataloader):
                variables = {'batch': batch}
                for fd in self.eval_forwards['dataset']:
                    if fd['mode'] == 'batch_forward':
                        variables[fd['out_name']] = fd['func'](
                            networks[fd['net_name']],
                            variables,
                            cuda_func=self.cuda_func,
                            **fd['param']
                        )
                    elif fd['mode'] == 'minning':
                        variables[fd['out_name']] = fd['func'](
                            variables,
                            **fd['param']
                        )

                errors.append(
                    loss_func.l1_modal_loss(
                        recc_acces(variables, self.eval_final_desc[0]),
                        recc_acces(variables, self.eval_final_desc[1]),
                        listed_maps=False
                    ).cpu().data[0]
                )

        return errors

    def _compute_sim(self, networks, queries, dataset):
        dataset_loader = utils.data.DataLoader(dataset, batch_size=1, num_workers=self.val_num_workers)
        queries_loader = utils.data.DataLoader(queries, batch_size=1, num_workers=self.val_num_workers)

        for network in networks.values():
            network.eval()

        dataset_feats = list()
        # Forward pass
        logger.info('Computing dataset feats')
        for batch in tqdm.tqdm(dataset_loader):
            variables = {'batch': batch}
            for fd in self.eval_forwards['dataset']:
                variables[fd['out_name']] = fd['func'](
                    networks[fd['net_name']],
                    variables,
                    cuda_func=self.cuda_func,
                    **fd['param']
                )
            final_desc = recc_acces(variables, self.eval_final_desc)
            dataset_feats.append((final_desc[0].cpu().data.numpy(), batch['coord'].cpu().numpy()))

        logger.info('Computing similarity')
        ranked = list()
        for query in tqdm.tqdm(queries_loader):
            variables = {'batch': query}
            for fd in self.eval_forwards['queries']:
                variables[fd['out_name']] = fd['func'](
                    networks[fd['net_name']],
                    variables,
                    cuda_func=self.cuda_func,
                    **fd['param']
                )
            feat = recc_acces(variables, self.eval_final_desc)[0].cpu().data.numpy()

            gt_pos = query['coord'].cpu().numpy()
            diff = [(np.dot(feat, d_feat[0]), np.linalg.norm(gt_pos - d_feat[1])) for d_feat in dataset_feats]
            sorted_index = list(np.argsort([d[0] for d in diff]))
            ranked.append([diff[idx][1] for idx in reversed(sorted_index)])

        return ranked

if __name__ == '__main__':
    logger.setLevel('INFO')
    modtouse = {'rgb': 'dataset.txt',
                'depth': 'mono_depth_dataset.txt'}
    transform = {
        'first': (tf.RandomResizedCrop(224),),
        'rgb': (tf.ToTensor(),),
        'depth': (tf.ToTensor(),)
    }
    transform_eval = {
        'first': (tf.Resize((224, 224)), tf.ToTensor(),),
    }

    query_data = Robotcar.VBLDataset(root=os.environ['ROBOTCAR'] + 'Robotcar_D1/Query/',
                                     modalities={'rgb': 'query.txt'},
                                     coord_file='coordxIm.txt',
                                     transform=transform_eval,
                                     bearing=False)
    data = Robotcar.VBLDataset(root=os.environ['ROBOTCAR'] + 'Robotcar_D1/Dataset/',
                               modalities={'rgb': 'dataset.txt'},
                               coord_file='coordxIm.txt',
                               transform=transform_eval,
                               bearing=False)

    dataset_1 = Robotcar.VBLDataset(root=os.environ['ROBOTCAR'] + 'training/TrainDataset_05_19_15/',
                                    modalities=modtouse,
                                    coord_file='coordxImbearing.txt',
                                    transform=transform)
    dataset_2 = Robotcar.VBLDataset(root=os.environ['ROBOTCAR'] + 'training/TrainDataset_08_28_15/',
                                    modalities=modtouse,
                                    coord_file='coordxImbearing.txt',
                                    transform=transform)
    dataset_3 = Robotcar.VBLDataset(root=os.environ['ROBOTCAR'] + 'training/TrainDataset_11_10_15/',
                                    modalities=modtouse,
                                    coord_file='coordxImbearing.txt',
                                    transform=transform)

    args = {
        'main': dataset_1,
        'examples': [dataset_2, dataset_3],
        'load_triplets': '200triplets.pth',
        'num_triplets': 100,
        'num_positives': 4,
        'num_negatives': 20
    }

    triplet_dataset = Robotcar.TripletDataset(**args)

    dtload = utils.data.DataLoader(triplet_dataset, batch_size=4)

    net = Desc.Main()
    trainer = Trainer(network=net,
                      cuda_on=True,
                      minning_func={'func': 'minning.no_selection',
                                    'param': {}},
                      triplet_loss={
                          'func': 'loss_func.adaptive_triplet_loss',
                          'param': dict()
                      }
                      )
    '''
    trainer.eval(dataset={'data': data, 'queries': query_data},
                 score_function=ScoreFunc.RecallAtN(n=1, radius=25),
                 ep=0)
    '''
    for b in tqdm.tqdm(dtload):
        trainer.train(b)
    trainer.eval(dataset={'data': data, 'queries': query_data},
                 score_function=ScoreFunc.RecallAtN(n=1, radius=25),
                 ep=1)
    """

    net = Desc.Deconv(end_relu=False, batch_norm=False)
    trainer = DeconvTrainer(network=net, cuda_on=True)
    '''
    trainer.eval(dataset={'data': data, 'queries': query_data},
                 score_function=ScoreFunc.RecallAtN(n=1, radius=25),
                 ep=0)
    '''
    for b in tqdm.tqdm(dtload):
        trainer.train(b)
    trainer.eval(dataset={'data': data, 'queries': query_data},
                 score_function=ScoreFunc.RecallAtN(n=1, radius=25),
                 ep=1)
    """