import setlog
import trainers.Base as Base
import torch.autograd as auto
import trainers.loss_functions as loss_func
import torch.utils as utils
import torch.utils.data
import os
import datasets.multmodtf as tf
import networks.Pose as Pose
import numpy as np
import tqdm
import copy
import score.Functions as ScoreFunc
import datasets.SevenScene as SevenScene
from trainers.minning_function import recc_acces
import pose_utils.BatchWrapper as b_wrapper
import trainers.minning_function as minning


logger = setlog.get_logger(__name__)


class Trainer(Base.BaseTrainer):
    def __init__(self, **kwargs):
        Base.BaseTrainer.__init__(
            self,
            lr=kwargs.pop('lr', 1e-5),
            momentum=kwargs.pop('momentum', 0.9),
            weight_decay=kwargs.pop('weight_decay', 0.001),
            cuda_on=kwargs.pop('cuda_on', True),
            optimizer_type=kwargs.pop('optimizer_type', 'ADAM'),
            network=kwargs.pop('network', None),
            val_num_workers=kwargs.pop('val_num_workers', 8)
        )

        self.pos_loss = kwargs.pop('pos_loss', loss_func.mean_dist)
        self.ori_loss = kwargs.pop('ori_loss', loss_func.mean_dist)
        self.combining_loss = kwargs.pop('combining_loss',
                                         loss_func.AlphaWeights(cuda=self.cuda_on))
        self.mod = kwargs.pop('mod', 'rgb')
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        self.optimizer = self.init_optimizer(
            self.network.get_training_layers() + self.combining_loss.params
        )
        self.loss_log['pos_loss'] = list()
        self.loss_log['ori_loss'] = list()
        self.loss_log['combined_loss'] = list()

    def train(self, batch):
        self.network.train()
        # Reset gradients
        self.optimizer.zero_grad()
        # Forward pass
        pose = self.network(auto.Variable(self.cuda_func(batch[self.mod]), requires_grad=True))
        gt_pos, gt_ori = (auto.Variable(self.cuda_func(batch['pose']['position'].float())),
                          auto.Variable(self.cuda_func(batch['pose']['orientation'].float())))

        pos_loss = self.pos_loss(pose['p'], gt_pos)
        ori_loss = self.ori_loss(pose['q'], gt_ori)
        loss = self.combining_loss.combine(pos_loss, ori_loss)

        loss.backward()  # calculate the gradients (backpropagation)
        self.optimizer.step()  # update the weights
        self.loss_log['pos_loss'].append(pos_loss.data[0])
        self.loss_log['ori_loss'].append(ori_loss.data[0])
        self.loss_log['combined_loss'].append(loss.data[0])
        logger.debug('Total loss is {}'.format(loss.data[0]))

    def eval(self, dataset, score_function, ep):
        if len(self.val_score) <= ep:
            errors = self._compute_errors(self.network, dataset)
            score = score_function(errors)
            self.val_score.append(score)
            if score_function.rank_score(score, self.best_net[0]):
                self.network.cpu()
                self.best_net = (score, copy.deepcopy(self.network.state_dict()))
                self.cuda_func(self.network)
        logger.info('Score is: {}'.format(self.val_score[ep]))

    def test(self, dataset, score_functions):
        net_to_test = copy.deepcopy(self.network)
        net_to_test.load_state_dict(self.best_net[1])
        errors = self._compute_errors(net_to_test, dataset)
        results = dict()
        for function_name, score_func in score_functions.items():
            results[function_name] = score_func(errors)
        return results

    def _compute_errors(self, network, dataset):
        network.eval()

        dataloader = utils.data.DataLoader(dataset,
                                           batch_size=1,
                                           num_workers=self.val_num_workers
                                           )
        errors = {
            'position': list(),
            'orientation': list()
        }

        logger.info('Computing position and orientation errors')
        for example in tqdm.tqdm(dataloader):
            pose = network(auto.Variable(self.cuda_func(example[self.mod]),
                                         requires_grad=False))
            errors['position'].append(np.linalg.norm(pose['p'].cpu().data.numpy() -
                                                     example['pose']['position'].numpy()))
            errors['orientation'].append(self.distance_between_q(pose['q'].cpu().data.numpy()[0],
                                                                 example['pose']['orientation'].numpy()[0]))
        return errors

    @staticmethod
    def distance_between_q(q1, q2):
        """
        Compute angle between 2 quaternions
        :param q1:
        :param q2:
        :return: angle (in degree)
        """
        # q2[0,1:] *= -1 # Inverse computation

        w3 = np.abs(np.dot(q1, q2))
        angle = 2 * np.arccos(w3)

        return np.rad2deg(angle)

    def serialize(self):
        ser = Base.BaseTrainer.serialize(self)
        self.combining_loss.cpu()
        ser['combining_loss'] = copy.deepcopy(self.combining_loss.state_directory())
        self.cuda_func(self.combining_loss)
        return ser

    def load(self, datas):
        self.combining_loss.cpu()
        self.network.cpu()

        self.optimizer.load_state_dict(datas['optimizer'])

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = self.cuda_func(v)

        self.combining_loss.load_state_directory(datas['combining_loss'])
        self.network.load_state_dict(datas['network'])
        self.best_net = datas['best_network']
        self.loss_log = datas['loss']
        self.val_score = datas['val_score']

        self.cuda_func(self.combining_loss)
        self.cuda_func(self.network)

        self.optimizer = self.init_optimizer(
            self.network.get_training_layers() + self.combining_loss.params
        )

        self.optimizer.load_state_dict(datas['optimizer'])

        for key, state in self.optimizer.state.items():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = self.cuda_func(v)


class Deconv(Trainer):
    def __init__(self, **kwargs):
        self.modal_loss = kwargs.pop('modal_loss', {'func': 'loss_func.l1_modal_loss',
                                                    'param': {'p': 1, 'factor': 1}})
        aux_loss = kwargs.pop('aux_loss', dict())
        self.aux_mod = kwargs.pop('aux_mod', 'mono_depth')

        Trainer.__init__(self, **kwargs)

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
        pose = self.network(auto.Variable(self.cuda_func(batch[self.mod]), requires_grad=True))
        gt_pos, gt_ori = (auto.Variable(self.cuda_func(batch['pose']['position'].float())),
                          auto.Variable(self.cuda_func(batch['pose']['orientation'].float())))
        gt_pose = {'p': gt_pos, 'q': gt_ori}

        gt_mod = auto.Variable(self.cuda_func(batch[self.aux_mod]), requires_grad=False)

        pos_loss = self.pos_loss(pose['full']['p'], gt_pos)
        ori_loss = self.ori_loss(pose['full']['q'], gt_ori)
        pose_loss = self.combining_loss.combine(pos_loss, ori_loss)

        modal_loss = self.modal_loss['func']((pose['maps'], ),
                                             (gt_mod, ),
                                             **self.modal_loss['param'])

        loss = pose_loss + modal_loss

        for name, aux_los in self.aux_loss.items():
            val = aux_los['func'](pose, gt_pose, combine_func=self.combining_loss.combine, **aux_los['param'])
            loss += val
            self.loss_log[name].append(val.data[0])
            logger.debug(name + ' loss is {}'.format(val.data[0]))

        loss.backward()  # calculate the gradients (backpropagation)
        self.optimizer.step()  # update the weights
        self.loss_log['pos_loss'].append(pos_loss.data[0])
        self.loss_log['ori_loss'].append(ori_loss.data[0])
        self.loss_log['combined_loss'].append(pose_loss.data[0])
        self.loss_log['modal_loss'].append(modal_loss.data[0])

        logger.debug('Total loss is {}'.format(loss.data[0]))


class MultNetTrainer(Base.BaseMultNetTrainer):
    """
    Pytorch 0.4
    """
    def __init__(self, **kwargs):
        training_pipeline = kwargs.pop('training_pipeline', list())
        eval_forwards = kwargs.pop('eval_forwards', dict())
        build_model_func = kwargs.pop('build_model_func', None)
        self.access_pose = kwargs.pop('access_pose', ['pose'])

        Base.BaseMultNetTrainer.__init__(self, **kwargs)

        self.training_pipeline = list()
        for action in training_pipeline:
            self.training_pipeline.append(action)
            if 'func' in self.training_pipeline[-1].keys():
                self.training_pipeline[-1]['func'] = eval(action['func'])
            if self.training_pipeline[-1]['mode'] == 'loss':
                self.loss_log[action['name']] = list()

        self.eval_forwards = {'data': list(), 'queries': list()}
        for forward in eval_forwards['data']:
            self.eval_forwards['data'].append(forward)
            self.eval_forwards['data'][-1]['func'] = eval(forward['func'])
        for forward in eval_forwards['queries']:
            self.eval_forwards['queries'].append(forward)
            self.eval_forwards['queries'][-1]['func'] = eval(forward['func'])

        self.build_model = eval(build_model_func)

    @property
    def device(self):
        return torch.device('cuda' if self.cuda_on else 'cpu')

    def train(self, batch):
        for network in self.networks.values():
            network = network.train().to(self.device)
            for params in network.get_training_layers():
                for param in params['params']:
                    param.requires_grad = True

        # Forward pass
        # TODO: .to to move on device
        variables = {'batch': batch}
        summed_loss = 0
        for action in self.training_pipeline:

            variables = self._sequential_forward(action, variables, self.networks)

            if action['mode'] == 'loss':
                input_args = [recc_acces(variables, name) for name in action['args']]
                val = action['func'](*input_args, **action['param'])
                summed_loss += val
                self.loss_log[action['name']].append(val.item())
                logger.debug(action['name'] + ' loss is {}'.format(val.item()))
            elif action['mode'] == 'backprop':
                self.optimizers[action['trainer']].zero_grad()

                summed_loss.backward()
                # summed_loss.backward(retain_graph=True)
                self.optimizers[action['trainer']].step()
                summed_loss = 0
                for name in self.optimizers_params[action['trainer']]['associated_net']:
                    for params in self.networks[name].get_training_layers():
                        for param in params['params']:
                            param.requires_grad = False
            elif action['mode'] == 'no_grad':
                for name in self.optimizers_params[action['trainer']]['associated_net']:
                    for params in self.networks[name].get_training_layers('all'):
                        for param in params['params']:
                            param.requires_grad = False
                for name in self.optimizers_params[action['trainer']]['associated_net']:
                    for params in self.networks[name].get_training_layers():
                        for param in params['params']:
                            param.requires_grad = True
            else:
                if action['mode'] not in ('batch_forward', 'forward', 'minning'):
                    raise NameError('Unknown action {}'.format(action['mode']))

        del summed_loss, variables

    def _sequential_forward(self, action, variables, networks):
        if action['mode'] == 'batch_forward':
            variables[action['out_name']] = action['func'](
                networks[action['net_name']],
                variables,
                cuda_func=self.cuda_func,
                **action['param']
            )
        elif action['mode'] == 'forward':
            variables[action['out_name']] = action['func'](
                networks[action['net_name']],
                variables,
                **action['param']
            )
        elif action['mode'] == 'minning':
            variables[action['out_name']] = action['func'](
                variables,
                **action['param']
            )

        return variables

    def eval(self, **kwargs):
        with torch.no_grad():
            dataset = kwargs.pop('dataset', None)
            score_function = kwargs.pop('score_function', None)
            ep = kwargs.pop('ep', None)
            if kwargs:
                logger.error('Unexpected **kwargs: %r' % kwargs)
                raise TypeError('Unexpected **kwargs: %r' % kwargs)

            if len(self.val_score) <= ep:
                if isinstance(score_function, ScoreFunc.Reconstruction_Error):
                    errors = self._compute_rerror(self.networks, dataset['queries'], dataset['data'])
                else:
                    errors = self._compute_errors(self.networks, dataset['queries'], dataset['data'])

                score = score_function(errors)

                self.val_score.append(score)
                if score_function.rank_score(score, self.best_net[0]):
                    self._save_current_net(score)

            logger.info('Score is: {}'.format(self.val_score[ep]))

    def test(self, **kwargs):
        with torch.no_grad():
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
                errors = self._compute_rerror(nets_to_test, dataset['queries'], dataset['data'])
            else:
                errors = self._compute_errors(nets_to_test, dataset['queries'], dataset['data'])
            results = dict()
            for function_name, score_func in score_functions.items():
                results[function_name] = score_func(errors)
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
                for action in self.eval_forwards['data']:
                    variables = self._sequential_forward(action, variables, networks)

                errors.append(
                    loss_func.l1_modal_loss(
                        recc_acces(variables, self.eval_final_desc[0]),
                        recc_acces(variables, self.eval_final_desc[1]),
                        listed_maps=False
                    ).cpu().data[0]
                )

        return errors

    def _compute_errors(self, networks, queries, dataset):
        for network in networks.values():
            network.eval()

        dataset_loader = utils.data.DataLoader(dataset, batch_size=1, num_workers=self.val_num_workers)
        queries_loader = utils.data.DataLoader(queries, batch_size=1, num_workers=self.val_num_workers)

        logger.info('Computing reference model')
        model = None
        for batch in tqdm.tqdm(dataset_loader):
            variables = {'batch': batch}
            for action in self.eval_forwards['data']:
                variables = self._sequential_forward(action, variables, networks)

            if model is None:
                model =  recc_acces(variables, ['model',])
            else:
                model = self.build_model((model, recc_acces(variables, ['model',])))

        errors = {
            'position': list(),
            'orientation': list()
        }
        logger.info('Computing position and orientation errors')
        for query in tqdm.tqdm(queries_loader):
            variables = {'batch': query}
            variables['model'] = model
            for action in self.eval_forwards['queries']:
                variables = self._sequential_forward(action, variables, networks)

            pose = recc_acces(variables, self.access_pose)

            errors['position'].append(np.linalg.norm(pose['p'].cpu().detach().numpy() -
                                                     query['pose']['position'].numpy()))
            errors['orientation'].append(self.distance_between_q(pose['q'].cpu().detach().numpy()[0],
                                                                 query['pose']['orientation'].numpy()[0]))
        return errors


    @staticmethod
    def distance_between_q(q1, q2):
        """
        Compute angle between 2 quaternions
        :param q1:
        :param q2:
        :return: angle (in degree)
        """
        #q2[1:] *= -1 # Inverse computation

        w3 = np.abs(np.dot(q1, q2))
        angle = 2 * np.arccos(w3)

        return np.rad2deg(angle)


if __name__ == '__main__':
    test_tf = {
            'first': (tf.Resize(240), tf.RandomResizedCrop(224),),
            'rgb': (tf.ColorJitter(), tf.ToTensor())
    }
    val_tf = {
        'first': (tf.Resize((224, 224)), ),
        'rgb': (tf.ToTensor(), )
    }
    root = os.environ['SEVENSCENES'] + 'heads/'

    train_dataset = SevenScene.Train(root=root,
                                     transform=test_tf,
                                     used_mod=('rgb',))

    val_dataset = SevenScene.Val(root=root,
                                 transform=val_tf,
                                 used_mod=('rgb',))

    dtload = utils.data.DataLoader(train_dataset, batch_size=24)

    net = Pose.Main()
    trainer = Trainer(network=net, cuda_on=False)
    trainer.eval(dataset=val_dataset,
                 score_function=ScoreFunc.GlobalPoseError(data_type='position',
                                                          pooling_type='mean'),
                 ep=0)
    for b in tqdm.tqdm(dtload):
        trainer.train(b)
        break
    trainer.eval(dataset=val_dataset,
                 score_function=ScoreFunc.GlobalPoseError(data_type='position',
                                                          pooling_type='mean'),
                 ep=1)
