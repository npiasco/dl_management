import setlog
import yaml
import torch.utils as utils
import torch.utils.data
import tqdm
import os
import copy
import networks.Descriptor
import trainer.TripletTrainers
import score.Functions
import datasets.Robotcar

logger = setlog.get_logger(__name__)


class Base:
    def __init__(self, **kwargs):
        self.root = kwargs.pop('root', None)
        self.param_file = kwargs.pop('dataset_file', 'params.yaml')
        self.trainer_file = kwargs.pop('trainer_file', 'trainer.yaml')
        self.cnn_file = kwargs.pop('cnn_type', 'cnn.yaml')
        if kwargs:
            logger.error('Unexpected **kwargs: %r' % kwargs)
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        with open(self.root + self.cnn_file, 'rt') as f:
            network_params = yaml.safe_load(f)
        self.network = eval(network_params['class'])(**network_params['param_class'])

        with open(self.root + self.trainer_file, 'rt') as f:
            trainer_params = yaml.safe_load(f)
        self.trainer = eval(trainer_params['class'])(network=self.network, **trainer_params['param_class'])
        self.eval_func = eval(trainer_params['eval_class'])(**trainer_params['param_eval_class'])
        self.test_func = dict()
        for test_func_name in trainer_params['test_func']:
            self.test_func[test_func_name] = eval(trainer_params['test_func'][test_func_name]['class'])\
                (
                    **trainer_params['test_func'][test_func_name]['param_class']
                )

        with open(self.root + self.param_file, 'rt') as f:
            params = yaml.safe_load(f)
        self.params = copy.deepcopy(params)
        self.curr_epoch = params.pop('curr_epoch', 0)
        self.max_epoch = params.pop('max_epoch', 1000)
        self.batch_size = params.pop('batch_size', 64)
        self.num_workers = params.pop('num_workers', 8)
        self.shuffle = params.pop('shuffle', True)
        self.results = params.pop('score', None)
        params.pop('saved_files')
        if params:
            logger.error('Unexpected **params: %r' % params)
            raise TypeError('Unexpected **params: %r' % params)

        if self.curr_epoch != 0:
            self.load()

    def train(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()

    def save(self, datas):
        self.params['saved_files'] = dict()
        for name, data in datas.items():
            self.params['saved_files'][name] = self.root + name + '.pth'
            torch.save(data, self.root + name + '.pth')

        self.params['curr_epoch'] = self.curr_epoch
        self.params['score'] = self.results
        with open(self.root + self.param_file, 'wt') as f:
            f.write(yaml.safe_dump(self.params))
        logger.info('Checkpoint saved at epoch {}'.format(self.curr_epoch))

    def load(self):
        datas = dict()
        for name, file in self.params['saved_files'].items():
            datas[name] = torch.load(file)
        self.trainer.load(datas)

    def print(self):
        raise NotImplementedError()


class DescriptorLearning(Base):
    def __init__(self, **kwargs):
        self.dataset_file = kwargs.pop('dataset_file', 'dataset.yaml')
        Base.__init__(self, **kwargs)

        with open(self.root + self.dataset_file, 'rt') as f:
            dataset_params = yaml.safe_load(f)

        self.data = dict()
        training_param = dict()
        training_param['main'] = self.creat_dataset(dataset_params['train']['param_class']['main'])
        dataset_params['train']['param_class'].pop('main')
        training_param['examples'] = [self.creat_dataset(d) for d in dataset_params['train']['param_class']['examples']]
        dataset_params['train']['param_class'].pop('examples')
        self.data['train'] = eval(dataset_params['train']['class'])(**training_param,
                                                                    **dataset_params['train']['param_class'])

        self.data['test'] = dict()
        self.data['test']['queries'] = self.creat_dataset(dataset_params['test']['queries'])
        self.data['test']['data'] = self.creat_dataset(dataset_params['test']['data'])

        self.data['val'] = dict()
        self.data['val']['queries'] = self.creat_dataset(dataset_params['val']['queries'])
        self.data['val']['data'] = self.creat_dataset(dataset_params['val']['data'])

        self.training_mod = dataset_params['training_mod']
        self.testing_mod = dataset_params['testing_mod']

    @staticmethod
    def creat_dataset(params):
        transform = dict()
        for name, content in params['param_class']['transform'].items():
            transform[name] = list()
            for diff_tf in content:
                if diff_tf['param_class']:
                    transform[name].append(eval(diff_tf['class'])(**diff_tf['param_class']))
                else:
                    transform[name].append(eval(diff_tf['class'])())
        params['param_class'].pop('transform')
        name = os.environ['ROBOTCAR'] + params['param_class']['root']
        params['param_class'].pop('root')
        return eval(params['class'])(root=name, transform=transform, **params['param_class'])


    def train(self):
        self.data['train'].used_mod = self.training_mod
        self.data['val']['queries'].used_mod = self.testing_mod
        self.data['val']['data'].used_mod = self.testing_mod

        logger.info('Initial saving berfore training.')
        self.save(self.trainer.serialize())

        logger.info('Getting the first validation score...')
        data_to_save = self.trainer.eval(self.data['val']['queries'],
                                         self.data['val']['data'],
                                         self.eval_func)
        dtload = utils.data.DataLoader(self.data['train'],
                                       batch_size=self.batch_size,
                                       num_workers=self.num_workers,
                                       shuffle=self.shuffle)

        for ep in tqdm.tqdm(range(self.curr_epoch, self.max_epoch)):
            for b in tqdm.tqdm(dtload):
                self.trainer.train(b)
            self.curr_epoch += 1
            data_to_save = self.trainer.eval(self.data['val']['queries'],
                                             self.data['val']['data'],
                                             self.eval_func,
                                             serialize=True)
            self.save(data_to_save)

    def test(self):
        self.data['test']['queries'].used_mod = self.testing_mod
        self.data['test']['data'].used_mod = self.testing_mod

        self.results = self.trainer.test(self.data['test']['queries'],
                                         self.data['test']['data'],
                                         self.test_func)


if __name__ == '__main__':
    sys = DescriptorLearning(root=os.environ['DATA'] + 'testing_sys/')
    sys.test()
    sys.train()
