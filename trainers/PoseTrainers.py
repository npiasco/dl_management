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
