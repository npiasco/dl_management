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

        self.training_mod = dataset_params['training_mod']
        self.testing_mod = dataset_params['testing_mod']

        self.network = eval(self.network_params['class'])(**self.network_params['param_class'])

        triplet_loss = self.trainer_params['param_class'].pop('triplet_loss',
                                                              None)
        minning_func = self.trainer_params['param_class'].pop('minning_func',
                                                              'trainers.minning_function.random')
        self.trainer = eval(self.trainer_params['class'])(network=self.network,
                                                          triplet_loss=triplet_loss,
                                                          minning_func=eval(minning_func),
                                                          **self.trainer_params['param_class'])
        if self.score_file is not None:
            self.load()

    def train(self):
        self.data['train'].used_mod = self.training_mod
        self.data['val']['queries'].used_mod = self.testing_mod
        self.data['val']['data'].used_mod = self.testing_mod
        BaseClass.Base.train(self)

    def test(self):
        self.data['test']['queries'].used_mod = self.testing_mod
        self.data['test']['data'].used_mod = self.testing_mod
        BaseClass.Base.test(self)

    def plot(self, **kwargs):
        BaseClass.Base.plot(self, **kwargs, size_dataset=len(self.data['train']))

    def print(self, dataset_name):
        if dataset_name == 'train':
            dtload = data.DataLoader(self.data[dataset_name], batch_size=4)
        elif dataset_name == 'val_query':
            dtload = data.DataLoader(self.data['val']['queries'], batch_size=16)
        elif dataset_name == 'val_data':
            dtload = data.DataLoader(self.data['val']['data'], batch_size=16)
        elif dataset_name == 'test_query':
            dtload = data.DataLoader(self.data['test']['queries'], batch_size=16)
        elif dataset_name == 'test_data':
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

        images_batch = torch.cat(buffer, 0)
        grid = torchvis.utils.make_grid(images_batch, nrow=4)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))


class Deconv(Default):
    def __init__(self, **kwargs):
        Default.__init__(self, **kwargs)
        if self.curr_epoch == 0:
            encoder_weight = self.network_params.get('encoder_weight', False)
            decoder_weight = self.network_params.get('decoder_weight', False)
            desc_weight = self.network_params.get('desc_weight', False)
            agg_weight = self.network_params.get('agg_weight', False)

            if encoder_weight:
                self.network.feature.load_state_dict(
                    torch.load(os.environ['CNN_WEIGHTS'] + encoder_weight)
                )
            if decoder_weight:
                self.network.deconv.load_state_dict(
                    torch.load(os.environ['CNN_WEIGHTS'] + decoder_weight)
                )
            if desc_weight:
                self.network.descriptor.load_state_dict(
                    torch.load(os.environ['CNN_WEIGHTS'] + desc_weight)
                )
            if agg_weight:
                self.network.feat_agg.load_state_dict(
                    torch.load(os.environ['CNN_WEIGHTS'] + agg_weight)
                )

    def map_print(self):
        self.network.train()  # To have the infered map
        self.data['train'].used_mod = self.training_mod
        dtload = data.DataLoader(self.data['train'], batch_size=4)
        plt.figure()
        ccmap = plt.get_cmap('jet', lut=1024)
        for b in dtload:
            modality = b['query'][self.trainer.aux_mod].contiguous().view(4, 1, 224, 224)
            output = self.network(
                auto.Variable(
                    self.trainer.cuda_func(
                        b['query'][self.trainer.mod]
                    ),
                    requires_grad=False
                )
            )

            images_batch = torch.cat((modality.cpu(), output['maps'].data.cpu()))
            grid = torchvis.utils.make_grid(images_batch, nrow=4)
            plt.imshow(grid.numpy().transpose(1, 2, 0)[:, :, 0], cmap=ccmap)
            plt.colorbar()
            plt.show()


if __name__ == '__main__':
    '''
    system = Default(root=os.environ['DATA'] + 'DescLearning/Template/')
    system.train()
    system.test()
    system.plot()
    system.print('val_data')
    '''
    system = Deconv(root=os.environ['DATA'] + 'DescLearning/DeconvTemplate/')
    system.map_print()
