import setlog
import trainer.Base as Base
import torch.nn.functional as func
import torch.autograd as auto
import trainer.minning_fonction as minning
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


class TripletTrainer(Base.BaseTrainer):
    def __init__(self, **kwargs):
        Base.BaseTrainer.__init__(
            self,
            lr=kwargs.pop('lr', 0.0001),
            momentum=kwargs.pop('momentum', 0.9),
            weight_decay=kwargs.pop('weight_decay', 0.001),
            cuda_on=kwargs.pop('cuda_on', True),
            optimizer_type=kwargs.pop('optimizer_type', 'SGD')
        )

        self.network = kwargs.pop('network', None)
        self.triplet_loss = kwargs.pop('triplet_loss', func.triplet_margin_loss)
        self.margin = kwargs.pop('margin', 0.25)
        self.minning_func = kwargs.pop('minning_func', minning.random)
        self.mod = kwargs.pop('mod', 'rgb')
        self.val_num_workers = kwargs.pop('val_num_workers', 8)
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        self.optimizer = self.init_optimizer(self.network.get_training_layers())
        self.loss_log = {
            'triplet_loss': list()
        }

        self.network.cpu()
        self.best_net = (0, copy.deepcopy(self.network.state_dict()))
        self.cuda_func(self.network)

    def train(self, batch):
        self.network.train()
        # Reset gradients
        self.optimizer.zero_grad()
        # dataset.associated_net = copy.deepcopy(net).cpu()

        # Forward pass
        anchor = self.network(self.cuda_func(auto.Variable(batch['query'][self.mod], requires_grad=True)))
        positive = self.minning_func(self, batch, 'positives')
        negative = self.minning_func(self, batch, 'negatives')

        loss = self.triplet_loss(anchor['desc'], positive['desc'], negative['desc'], margin=self.margin)

        loss.backward()  # calculate the gradients (backpropagation)
        self.optimizer.step()  # update the weights
        self.loss_log['triplet_loss'].append(loss.data[0])
        logger.debug('Triplet loss is {}'.format(loss.data[0]))

    def eval(self, queries, dataset, score_function):
        dataset.used_mod = [self.mod]
        queries.used_mod = [self.mod]
        dataset_loader =  utils.data.DataLoader(dataset, batch_size=1, num_workers=self.val_num_workers)
        queries_loader = utils.data.DataLoader(queries, batch_size=1, num_workers=self.val_num_workers)

        self.network.eval()

        logger.info('Computing dataset feats')
        dataset_feats = [(self.network(self.cuda_func(auto.Variable(example[self.mod])))[0].cpu().data.numpy(),
                          example['coord'].cpu().numpy())
                         for example in tqdm.tqdm(dataset_loader)]

        logger.info('Computing similarity')
        ranked = list()
        for query in tqdm.tqdm(queries_loader):
            feat = self.network(self.cuda_func(auto.Variable(query[self.mod])))[0].cpu().data.numpy()
            gt_pos = query['coord'].cpu().numpy()
            diff = [(np.dot(feat, d_feat[0]), np.linalg.norm(gt_pos - d_feat[1])) for d_feat in dataset_feats]
            sorted_index = list(np.argsort([d[0] for d in diff]))
            ranked.append([diff[idx][1] for idx in reversed(sorted_index)])

        score = score_function(ranked)
        logger.info('Score is: {}'.format(score))
        if score_function.rank_score(score, self.best_net[0]):
            self.network.cpu()
            self.best_net = (score, copy.deepcopy(self.network.state_dict()))
            self.cuda_func(self.network)

    def serialize(self):
        raise NotImplementedError()


if __name__ == '__main__':
    logger.setLevel('INFO')
    modtouse = {'rgb': 'dataset.txt'}
    transform = {
        'first': (tf.RandomResizedCrop(224),),
        'rgb': (tf.ToTensor(), tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
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

    triplet_dataset = Robotcar.TripletDataset(dataset_1, dataset_2, dataset_3,
                                              num_triplets=100, num_positives=2, num_negative=20)
    dtload = utils.data.DataLoader(triplet_dataset, batch_size=4)

    network = Desc.Main(end_relu=True, batch_norm=False)
    trainer = TripletTrainer(network=network, cuda_on=False)
    trainer.eval(query_data, data, ScoreFunc.RecallAtN(n=1, radius=25))
    for b in tqdm.tqdm(dtload):
        trainer.train(b)
    trainer.eval(query_data, data, ScoreFunc.RecallAtN(n=1, radius=25))
