import setlog
import trainer.Base as Base
import torch.nn.functional as func
import torch.autograd as auto
import trainer.minning_fonction as minning
import datasets.Robotcar as Robotcar
import torch.utils as utils
import os
import datasets.multmodtf as tf
import networks.Descriptor as Desc
import tqdm


logger = setlog.get_logger(__name__)


class TripletTrainer(Base.BaseTrainer):
    def __init__(self, **kwargs):
        Base.BaseTrainer.__init__(
            self,
            batch_size=kwargs.pop('batch_size', 5),
            max_epoch=kwargs.pop('max_epoch ', 20),
            lr=kwargs.pop('lr', 0.0001),
            momentum=kwargs.pop('momentum', 0.9),
            weight_decay=kwargs.pop('weight_decay', 0.001),
            shuffle=kwargs.pop('shuffle', True),
            cuda_on=kwargs.pop('cuda_on', True),
            optimizer_type=kwargs.pop('optimizer_type', 'SGD')
        )

        self.network = kwargs.pop('network', None)
        self.triplet_loss = kwargs.pop('triplet_loss', func.triplet_margin_loss)
        self.margin = kwargs.pop('margin', 0.25)
        self.minning_func = kwargs.pop('minning_func', minning.random)
        self.mod = kwargs.pop('mod', 'rgb')
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        self.optimizer = self.init_optimizer(self.network.get_training_layers())
        self.loss_log = {
            'triplet_loss': list()
        }


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

    def eval(self):
        raise NotImplementedError()

    def serialize(self):
        raise NotImplementedError()

if __name__=='__main__':
    logger.setLevel('INFO')
    modtouse = {'rgb': 'dataset.txt'}
    transform = {
        'first': (tf.RandomResizedCrop(224),),
        'rgb': (tf.ToTensor(), tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])),
    }

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
    print(len(dataset_1), len(dataset_2),len(dataset_3))

    dataset_1[len(dataset_1) - 1]
    dataset_2[len(dataset_2) - 2]
    dataset_3[len(dataset_3) - 2]

    triplet_dataset = Robotcar.TripletDataset(dataset_1, dataset_2, dataset_3,
                                     num_triplets=100, num_positives=2, num_negative=20)
    dtload = utils.data.DataLoader(triplet_dataset, batch_size=4)

    network = Desc.Main().cuda()
    trainer = TripletTrainer(network=network)
    for batch in tqdm.tqdm(dtload):
        trainer.train(batch)


